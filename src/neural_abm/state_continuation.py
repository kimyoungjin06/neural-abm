"""Shared state-continuation objective helpers for binary neural toys."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


StateContinuationObjectiveProfile = Literal[
    "material",
    "custom",
    "linear_balanced",
    "linear_welfare_heavy",
    "nonlinear_mild",
    "nonlinear_interaction",
]
StateContinuationObjectiveActivation = Literal["identity", "tanh"]
DomainBootstrapTeacher = Literal["reputation_imitation"]
DomainBootstrapDecay = Literal["none", "linear"]
DomainDistillBootstrapLoss = Literal["bce", "kl"]
DomainDistillBootstrapScope = Literal["all", "revised"]
DomainDecisionReplaySuccess = Literal["teacher_objective_postsocial"]
BasinCreditCritic = Literal["prototype_phase", "contrastive_phase"]
BasinCreditMethod = Literal["one_step_ablation"]
BasinCreditTarget = Literal["ceiling"]
BasinCreditTrainingScope = Literal["revised", "all"]
BasinCreditTrainingPassSchedule = Literal[
    "fixed",
    "target_score_decay",
    "credit_signal_escalation",
]
BasinCreditLearnedFallback = Literal["prototype", "zero"]
BasinCreditLearnedReplaySelection = Literal[
    "all",
    "confident",
    "confident_agreement",
    "confident_disagreement",
]
BasinCreditLearnedReplayFloorSource = Literal["prototype_abs", "learned_abs"]
BasinCreditLearnedReplayFloorSchedule = Literal["fixed", "linear_decay"]
BasinCreditLearnedReplayMode = Literal["hard", "soft_attention", "learned_weight"]


BASIN_PHASE_EMBEDDING_DIM = 5


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "material": {
        "material_weight": 1.0,
        "social_weight": 0.0,
        "welfare_weight": 0.0,
        "environment_weight": 0.0,
        "risk_weight": 0.0,
        "activation": "identity",
        "activation_scale": 1.0,
        "social_welfare_interaction_weight": 0.0,
    },
    "linear_balanced": {
        "material_weight": 1.0,
        "social_weight": 1.0,
        "welfare_weight": 1.0,
        "environment_weight": 0.0,
        "risk_weight": 0.0,
        "activation": "identity",
        "activation_scale": 1.0,
        "social_welfare_interaction_weight": 0.0,
    },
    "linear_welfare_heavy": {
        "material_weight": 1.0,
        "social_weight": 0.5,
        "welfare_weight": 2.0,
        "environment_weight": 0.0,
        "risk_weight": 0.0,
        "activation": "identity",
        "activation_scale": 1.0,
        "social_welfare_interaction_weight": 0.0,
    },
    "nonlinear_mild": {
        "material_weight": 1.0,
        "social_weight": 1.0,
        "welfare_weight": 1.0,
        "environment_weight": 0.0,
        "risk_weight": 0.0,
        "activation": "tanh",
        "activation_scale": 1.0,
        "social_welfare_interaction_weight": 0.0,
    },
    "nonlinear_interaction": {
        "material_weight": 1.0,
        "social_weight": 1.0,
        "welfare_weight": 1.0,
        "environment_weight": 0.0,
        "risk_weight": 0.0,
        "activation": "tanh",
        "activation_scale": 1.0,
        "social_welfare_interaction_weight": 1.0,
    },
}


class StateContinuationObjectiveConfig(BaseModel):
    """Weights for turning domain continuation signals into policy advantages."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["material", "state_continuation"] = "material"
    profile: StateContinuationObjectiveProfile = "material"
    material_weight: float = 1.0
    social_weight: float = 0.0
    welfare_weight: float = 0.0
    environment_weight: float = 0.0
    risk_weight: float = 0.0
    activation: StateContinuationObjectiveActivation = "identity"
    activation_scale: float = Field(default=1.0, gt=0.0)
    social_welfare_interaction_weight: float = 0.0
    normalize: Literal["payoff_scale", "unit"] = "payoff_scale"
    clip_abs: float | None = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def resolve_profile_defaults(self) -> "StateContinuationObjectiveConfig":
        fields_set = self.model_fields_set
        if "profile" not in fields_set and self.mode == "state_continuation":
            self.profile = "custom"
        elif self.profile != "material":
            self.mode = "state_continuation"
        elif "profile" in fields_set:
            self.mode = "material"

        profile_defaults = _PROFILE_DEFAULTS.get(self.profile)
        if profile_defaults is not None:
            for field_name, value in profile_defaults.items():
                setattr(self, field_name, value)
        return self

    def uses_state_continuation(self) -> bool:
        return self.mode == "state_continuation"


class DomainBootstrapConfig(BaseModel):
    """Early-phase teacher signal blended into a state-continuation objective."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    teacher: DomainBootstrapTeacher = "reputation_imitation"
    epochs: int = Field(default=5, gt=0)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    teacher_scale: float = Field(default=1.0, ge=0.0)
    decay: DomainBootstrapDecay = "linear"
    decision_enabled: bool = False
    decision_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    decision_epochs: int = Field(default=5, gt=0)
    decision_decay: DomainBootstrapDecay = "linear"
    distill_enabled: bool = False
    distill_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    distill_epochs: int = Field(default=5, gt=0)
    distill_decay: DomainBootstrapDecay = "linear"
    distill_loss: DomainDistillBootstrapLoss = "bce"
    distill_scope: DomainDistillBootstrapScope = "all"
    distill_teacher: DomainBootstrapTeacher = "reputation_imitation"
    distill_stable_teacher_only: bool = False
    distill_teacher_margin_min: float = Field(default=0.0, ge=0.0, le=0.5)
    distill_gradient_gate_enabled: bool = False
    distill_gradient_min_cosine: float = Field(default=0.0, ge=-1.0, le=1.0)
    replay_enabled: bool = False
    replay_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    replay_epochs: int = Field(default=5, gt=0)
    replay_decay: DomainBootstrapDecay = "linear"
    replay_teacher: DomainBootstrapTeacher = "reputation_imitation"
    replay_success: DomainDecisionReplaySuccess = "teacher_objective_postsocial"
    replay_stable_teacher_only: bool = True
    replay_teacher_margin_min: float = Field(default=0.0, ge=0.0, le=0.5)
    replay_require_objective_agreement: bool = True
    replay_require_postsocial_alignment_improvement: bool = True


class BasinCreditConfig(BaseModel):
    """Counterfactual basin-credit objective configuration.

    The v1 implementation supports one-step ablation with prototype phase
    scoring only. ``contrastive_phase`` is reserved for a later learned critic.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    critic: BasinCreditCritic = "prototype_phase"
    credit_method: BasinCreditMethod = "one_step_ablation"
    objective_weight: float = Field(default=0.0, ge=0.0)
    individual_weight: float = Field(default=0.0, ge=0.0)
    local_social_weight: float = Field(default=0.0, ge=0.0)
    basin_weight: float = Field(default=1.0, ge=0.0)
    training_scope: BasinCreditTrainingScope = "revised"
    training_passes: int = Field(default=1, gt=0)
    training_pass_schedule: BasinCreditTrainingPassSchedule = "fixed"
    min_training_passes: int = Field(default=1, gt=0)
    training_pass_score_threshold: float = Field(default=0.995, ge=-1.0, le=1.0)
    training_pass_credit_positive_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )
    training_pass_credit_delta_threshold: float = 0.0
    horizon: int = Field(default=1, gt=0)
    target_basin: BasinCreditTarget = "ceiling"
    critic_temperature: float = Field(default=1.0, gt=0.0)
    prototype_decay: float = Field(default=0.90, ge=0.0, lt=1.0)
    learned_diagnostic_enabled: bool = False
    learned_diagnostic_model_path: Path | None = None
    learned_diagnostic_abstention_margin_threshold: float = Field(
        default=0.005,
        ge=0.0,
    )
    learned_diagnostic_uncertainty_threshold: float = Field(default=0.05, ge=0.0)
    learned_credit_enabled: bool = False
    learned_credit_model_path: Path | None = None
    learned_credit_abstention_margin_threshold: float = Field(default=0.005, ge=0.0)
    learned_credit_uncertainty_threshold: float = Field(default=0.05, ge=0.0)
    learned_credit_fallback: BasinCreditLearnedFallback = "prototype"
    learned_credit_replay_selection: BasinCreditLearnedReplaySelection = "all"
    learned_credit_replay_mode: BasinCreditLearnedReplayMode = "hard"
    learned_credit_replay_weight_model_path: Path | None = None
    learned_credit_replay_min_selected_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    learned_credit_replay_floor_source: BasinCreditLearnedReplayFloorSource = (
        "prototype_abs"
    )
    learned_credit_replay_floor_schedule: BasinCreditLearnedReplayFloorSchedule = (
        "fixed"
    )
    learned_credit_replay_floor_start_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    learned_credit_replay_floor_decay_epochs: int = Field(default=1, gt=0)
    learned_credit_replay_soft_min_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    learned_credit_replay_soft_disagreement_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_v1_slice(self) -> "BasinCreditConfig":
        if self.horizon != 1:
            raise ValueError("Basin-credit v1 only supports horizon=1")
        if self.enabled and self.critic == "contrastive_phase":
            raise ValueError(
                "Basin-credit critic='contrastive_phase' is reserved until "
                "the learned contrastive critic is implemented; use "
                "critic='prototype_phase' for the v1 scaffold"
            )
        if self.min_training_passes > self.training_passes:
            raise ValueError(
                "Basin-credit min_training_passes must be <= training_passes"
            )
        if self.learned_diagnostic_enabled and self.learned_diagnostic_model_path is None:
            raise ValueError(
                "Basin-credit learned diagnostic requires "
                "learned_diagnostic_model_path"
            )
        if self.learned_credit_enabled and (
            self.learned_credit_model_path is None
            and self.learned_diagnostic_model_path is None
        ):
            raise ValueError(
                "Basin-credit learned credit requires learned_credit_model_path "
                "or learned_diagnostic_model_path"
            )
        if (
            self.learned_credit_enabled
            and self.learned_credit_replay_mode == "learned_weight"
            and self.learned_credit_replay_weight_model_path is None
        ):
            raise ValueError(
                "Basin-credit learned_weight replay requires "
                "learned_credit_replay_weight_model_path"
            )
        return self


@dataclass(frozen=True)
class StateContinuationComponents:
    """Vectorized component advantages and their combined policy signal."""

    material: np.ndarray
    social: np.ndarray
    welfare: np.ndarray
    environment: np.ndarray
    risk: np.ndarray
    linear: np.ndarray
    interaction: np.ndarray
    activation_input: np.ndarray
    effective: np.ndarray
    objective_profile: str

    def with_effective(self, effective: np.ndarray) -> "StateContinuationComponents":
        return StateContinuationComponents(
            material=self.material,
            social=self.social,
            welfare=self.welfare,
            environment=self.environment,
            risk=self.risk,
            linear=self.linear,
            interaction=self.interaction,
            activation_input=self.activation_input,
            effective=effective,
            objective_profile=self.objective_profile,
        )


@dataclass(frozen=True)
class DomainBootstrapDiagnostics:
    """Teacher-bootstrap arrays and scalar schedule used for logging."""

    weight: float
    teacher_signed: np.ndarray
    bootstrapped_effective: np.ndarray
    teacher: str


@dataclass(frozen=True)
class DomainDecisionBootstrapDiagnostics:
    """Teacher decision-probability arrays and scalar schedule used for logging."""

    weight: float
    teacher_probabilities: np.ndarray
    neural_probabilities: np.ndarray
    bootstrapped_probabilities: np.ndarray
    teacher: str


@dataclass(frozen=True)
class DomainDecisionReplayDiagnostics:
    """Teacher-scaffold replay gates and scalar schedule used for logging."""

    weight: float
    teacher_probabilities: np.ndarray
    replay_actions: np.ndarray
    candidate_mask: np.ndarray
    stable_teacher_mask: np.ndarray
    objective_agreement_mask: np.ndarray
    postsocial_improvement_mask: np.ndarray
    applied_mask: np.ndarray
    teacher: str


@dataclass(frozen=True)
class DomainDistillBootstrapDiagnostics:
    """Teacher distillation arrays and scalar schedule used for logging."""

    weight: float
    teacher_probabilities: np.ndarray
    neural_probabilities: np.ndarray
    bce: np.ndarray
    kl: np.ndarray
    argmax_agreement: np.ndarray
    realized_action_agreement: np.ndarray
    teacher: str


@dataclass(frozen=True)
class DomainTeacherAlignmentDiagnostics:
    """Teacher/neural alignment arrays across the local and social update path."""

    teacher: str
    teacher_pre_action: np.ndarray
    teacher_post_action: np.ndarray
    pre_local: np.ndarray
    post_local: np.ndarray
    post_social: np.ndarray
    realized_actions: np.ndarray
    objective_effective: np.ndarray
    base_losses: np.ndarray
    distill_losses: np.ndarray
    base_grad_norms: np.ndarray
    distill_grad_norms: np.ndarray
    grad_cosines: np.ndarray
    distill_candidate_mask: np.ndarray
    distill_stable_teacher_mask: np.ndarray
    distill_gradient_gate_mask: np.ndarray
    distill_applied_mask: np.ndarray


@dataclass(frozen=True)
class BasinCriticOutput:
    """Phase-critic outputs for one or more basin-state embeddings."""

    basin_embedding: np.ndarray
    target_basin_score: np.ndarray
    non_target_basin_score: np.ndarray
    phase_confidence: np.ndarray


@dataclass
class PrototypePhaseBasinCritic:
    """Online prototype phase scorer used by the v1 basin-credit scaffold."""

    target_prototype: np.ndarray = field(default_factory=lambda: np.asarray(
        [1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    ))
    non_target_prototype: np.ndarray = field(default_factory=lambda: np.asarray(
        [0.0, 0.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    ))

    def evaluate(
        self,
        basin_embedding: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> BasinCriticOutput:
        embedding = _as_basin_embedding_batch(basin_embedding)
        target_score = _cosine_similarity_rows(embedding, self.target_prototype)
        non_target_score = _cosine_similarity_rows(
            embedding,
            self.non_target_prototype,
        )
        confidence = _sigmoid((target_score - non_target_score) / temperature)
        return BasinCriticOutput(
            basin_embedding=embedding,
            target_basin_score=target_score,
            non_target_basin_score=non_target_score,
            phase_confidence=confidence,
        )

    def update(
        self,
        basin_embedding: np.ndarray,
        *,
        target_reached: bool,
        decay: float,
    ) -> None:
        embedding = _as_basin_embedding_batch(basin_embedding)
        prototype = np.mean(embedding, axis=0)
        if target_reached:
            self.target_prototype = decay * self.target_prototype + (
                1.0 - decay
            ) * prototype
        else:
            self.non_target_prototype = decay * self.non_target_prototype + (
                1.0 - decay
            ) * prototype


ContrastivePhaseBasinCritic = PrototypePhaseBasinCritic


@dataclass(frozen=True)
class BasinCreditDiagnostics:
    """Per-agent one-step basin credit and associated phase-critic scores."""

    weight: float
    objective_weight: float
    individual_weight: float
    local_social_weight: float
    training_scope: str
    training_pass_schedule: str
    training_passes: int
    configured_training_passes: int
    min_training_passes: int
    training_pass_score_threshold: float
    training_pass_credit_positive_threshold: float
    training_pass_credit_delta_threshold: float
    target: str
    selected_action_credit: np.ndarray
    score_observed: np.ndarray
    score_counterfactual: np.ndarray
    applied_mask: np.ndarray
    phase_confidence: np.ndarray
    critic: str
    credit_method: str


_DOMAIN_DECISION_REPLAY_EMPTY_FIELDS: dict[str, object] = {
    "domain_decision_replay_weight": "",
    "domain_decision_replay_candidate_rate": "",
    "domain_decision_replay_applied_rate": "",
    "domain_decision_replay_rejected_unstable_teacher_rate": "",
    "domain_decision_replay_rejected_objective_rate": "",
    "domain_decision_replay_rejected_postsocial_rate": "",
    "domain_decision_replay_teacher_probability_mean": "",
    "domain_decision_replay_teacher": "",
}

_DOMAIN_BASIN_CREDIT_EMPTY_FIELDS: dict[str, object] = {
    "domain_basin_training_scope": "",
    "domain_basin_training_pass_schedule": "",
    "domain_basin_training_passes": "",
    "domain_basin_training_passes_configured": "",
    "domain_basin_min_training_passes": "",
    "domain_basin_training_pass_score_threshold": "",
    "domain_basin_training_pass_credit_positive_threshold": "",
    "domain_basin_training_pass_credit_delta_threshold": "",
    "domain_basin_training_candidate_rate": "",
    "domain_basin_objective_weight": "",
    "domain_basin_individual_weight": "",
    "domain_basin_local_social_weight": "",
    "domain_basin_credit_weight": "",
    "domain_basin_score_mean": "",
    "domain_basin_score_delta_mean": "",
    "domain_basin_credit_positive_rate": "",
    "domain_basin_phase_confidence_mean": "",
    "domain_basin_target": "",
}

_DOMAIN_BASIN_CREDIT_TRAINING_EMPTY_FIELDS: dict[str, object] = {
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

_DOMAIN_BASIN_CREDIT_MICRO_EMPTY_FIELDS: dict[str, object] = {
    "domain_basin_credit": "",
    "domain_basin_score_observed": "",
    "domain_basin_score_counterfactual": "",
    "domain_basin_credit_applied": "",
}

_DOMAIN_BASIN_CREDIT_TRAINING_MICRO_EMPTY_FIELDS: dict[str, object] = {
    "domain_basin_training_credit_source": "",
    "domain_basin_training_replay_selection": "",
    "domain_basin_training_replay_selected": "",
    "domain_basin_training_replay_weight": "",
    "domain_basin_training_learned_credit_used": "",
    "domain_basin_action1_advantage": "",
    "domain_basin_training_effective_advantage": "",
}


_DOMAIN_TEACHER_ALIGNMENT_EMPTY_FIELDS: dict[str, object] = {
    "domain_teacher_policy_bce_pre_local_mean": "",
    "domain_teacher_policy_bce_post_local_mean": "",
    "domain_teacher_policy_bce_post_social_mean": "",
    "domain_teacher_policy_kl_pre_local_mean": "",
    "domain_teacher_policy_kl_post_local_mean": "",
    "domain_teacher_policy_kl_post_social_mean": "",
    "domain_teacher_bce_delta_local": "",
    "domain_teacher_bce_delta_social": "",
    "domain_teacher_neural_argmax_agreement_pre_local": "",
    "domain_teacher_neural_argmax_agreement_post_local": "",
    "domain_teacher_neural_argmax_agreement_post_social": "",
    "domain_teacher_realized_action_agreement": "",
    "domain_teacher_probability_pre_action_mean": "",
    "domain_teacher_probability_post_action_mean": "",
    "domain_teacher_target_shift_mean": "",
    "domain_teacher_target_flip_rate": "",
    "domain_effective_advantage_teacher_sign_agreement": "",
    "domain_effective_advantage_teacher_sign_conflict_rate": "",
    "domain_effective_advantage_teacher_margin_mean": "",
    "domain_base_loss_mean": "",
    "domain_distill_loss_mean": "",
    "domain_base_grad_norm_mean": "",
    "domain_distill_grad_norm_mean": "",
    "domain_base_distill_grad_cosine_mean": "",
    "domain_base_distill_grad_cosine_negative_rate": "",
    "domain_distill_candidate_rate": "",
    "domain_distill_applied_rate": "",
    "domain_distill_rejected_unstable_teacher_rate": "",
    "domain_distill_rejected_gradient_rate": "",
    "domain_teacher_alignment_teacher": "",
}


def _component_or_zero(
    values: np.ndarray | None,
    *,
    like: np.ndarray,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.zeros_like(like, dtype=np.float64)
    component = np.asarray(values, dtype=np.float64)
    if component.shape != like.shape:
        raise ValueError(
            f"{name} advantage shape {component.shape} does not match "
            f"material shape {like.shape}"
        )
    return component


def combine_state_continuation_advantages(
    *,
    material: np.ndarray,
    objective: StateContinuationObjectiveConfig,
    social: np.ndarray | None = None,
    welfare: np.ndarray | None = None,
    environment: np.ndarray | None = None,
    risk: np.ndarray | None = None,
) -> StateContinuationComponents:
    """Combine material and continuation components into an effective advantage."""

    material_values = np.asarray(material, dtype=np.float64)
    social_values = _component_or_zero(social, like=material_values, name="social")
    welfare_values = _component_or_zero(welfare, like=material_values, name="welfare")
    environment_values = _component_or_zero(
        environment,
        like=material_values,
        name="environment",
    )
    risk_values = _component_or_zero(risk, like=material_values, name="risk")

    linear = (
        objective.material_weight * material_values
        + objective.social_weight * social_values
        + objective.welfare_weight * welfare_values
        + objective.environment_weight * environment_values
        - objective.risk_weight * risk_values
    )
    interaction = (
        objective.social_welfare_interaction_weight * social_values * welfare_values
    )
    activation_input = linear + interaction
    effective = _apply_activation(
        activation_input,
        activation=objective.activation,
        scale=objective.activation_scale,
    )
    if objective.uses_state_continuation():
        if objective.clip_abs is not None:
            effective = np.clip(effective, -objective.clip_abs, objective.clip_abs)

    return StateContinuationComponents(
        material=material_values,
        social=social_values,
        welfare=welfare_values,
        environment=environment_values,
        risk=risk_values,
        linear=linear,
        interaction=interaction,
        activation_input=activation_input,
        effective=effective,
        objective_profile=objective.profile,
    )


def _apply_activation(
    values: np.ndarray,
    *,
    activation: StateContinuationObjectiveActivation,
    scale: float,
) -> np.ndarray:
    if activation == "identity":
        return values.copy()
    if activation == "tanh":
        return scale * np.tanh(values / scale)
    raise ValueError(f"Unsupported state-continuation activation: {activation}")


def _scheduled_bootstrap_weight(
    *,
    enabled: bool,
    weight: float,
    epochs: int,
    decay: DomainBootstrapDecay,
    epoch: int,
) -> float:
    if not enabled:
        return 0.0
    phase_epoch = max(int(epoch), 1)
    if phase_epoch > epochs:
        return 0.0
    if decay == "none":
        return float(weight)
    if decay == "linear":
        if epochs == 1:
            return float(weight)
        progress = (phase_epoch - 1) / float(epochs - 1)
        return float(weight) * max(0.0, 1.0 - progress)
    raise ValueError(f"Unsupported domain bootstrap decay: {decay}")


def domain_bootstrap_weight(
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> float:
    """Return the 1-based epoch target-bootstrap weight."""

    return _scheduled_bootstrap_weight(
        enabled=bootstrap.enabled,
        weight=bootstrap.weight,
        epochs=bootstrap.epochs,
        decay=bootstrap.decay,
        epoch=epoch,
    )


def domain_decision_bootstrap_weight(
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> float:
    """Return the 1-based epoch decision-bootstrap weight."""

    return _scheduled_bootstrap_weight(
        enabled=bootstrap.decision_enabled,
        weight=bootstrap.decision_weight,
        epochs=bootstrap.decision_epochs,
        decay=bootstrap.decision_decay,
        epoch=epoch,
    )


def domain_distill_bootstrap_weight(
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> float:
    """Return the 1-based epoch policy-distillation bootstrap weight."""

    return _scheduled_bootstrap_weight(
        enabled=bootstrap.distill_enabled,
        weight=bootstrap.distill_weight,
        epochs=bootstrap.distill_epochs,
        decay=bootstrap.distill_decay,
        epoch=epoch,
    )


def domain_decision_replay_weight(
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> float:
    """Return the 1-based epoch teacher-scaffold replay weight."""

    return _scheduled_bootstrap_weight(
        enabled=bootstrap.replay_enabled,
        weight=bootstrap.replay_weight,
        epochs=bootstrap.replay_epochs,
        decay=bootstrap.replay_decay,
        epoch=epoch,
    )


def teacher_probabilities_to_signed_advantages(
    teacher_probabilities: np.ndarray,
    *,
    teacher_scale: float,
) -> np.ndarray:
    """Map teacher cooperation probabilities from [0, 1] to signed advantages."""

    probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
    return float(teacher_scale) * (2.0 * probabilities - 1.0)


def blend_bootstrap_signed_advantages(
    objective_signed: np.ndarray,
    teacher_signed: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend objective and teacher signed advantages with an exact convex formula."""

    objective_values = np.asarray(objective_signed, dtype=np.float64)
    teacher_values = np.asarray(teacher_signed, dtype=np.float64)
    if objective_values.shape != teacher_values.shape:
        raise ValueError(
            "teacher signed advantage shape "
            f"{teacher_values.shape} does not match objective shape "
            f"{objective_values.shape}"
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"domain bootstrap weight must be in [0, 1]; got {weight:g}")
    return (1.0 - weight) * objective_values + weight * teacher_values


def blend_bootstrap_decision_probabilities(
    neural_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend neural and teacher cooperation probabilities exactly convexly."""

    neural_values = np.asarray(neural_probabilities, dtype=np.float64)
    teacher_values = np.asarray(teacher_probabilities, dtype=np.float64)
    if neural_values.shape != teacher_values.shape:
        raise ValueError(
            "teacher decision probability shape "
            f"{teacher_values.shape} does not match neural decision probability "
            f"shape {neural_values.shape}"
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            f"domain decision bootstrap weight must be in [0, 1]; got {weight:g}"
        )
    return (1.0 - weight) * neural_values + weight * teacher_values


def teacher_policy_bce(
    neural_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
) -> np.ndarray:
    """Return Bernoulli cross entropy from teacher probabilities to neural probs."""

    neural_values, teacher_values = _aligned_probability_arrays(
        neural_probabilities,
        teacher_probabilities,
    )
    eps = np.finfo(np.float64).eps
    neural_values = np.clip(neural_values, eps, 1.0 - eps)
    return -(
        teacher_values * np.log(neural_values)
        + (1.0 - teacher_values) * np.log(1.0 - neural_values)
    )


def teacher_policy_kl(
    neural_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
) -> np.ndarray:
    """Return Bernoulli KL divergence KL(teacher || neural)."""

    neural_values, teacher_values = _aligned_probability_arrays(
        neural_probabilities,
        teacher_probabilities,
    )
    eps = np.finfo(np.float64).eps
    neural_values = np.clip(neural_values, eps, 1.0 - eps)
    teacher_values = np.clip(teacher_values, eps, 1.0 - eps)
    return (
        teacher_values * (np.log(teacher_values) - np.log(neural_values))
        + (1.0 - teacher_values)
        * (np.log(1.0 - teacher_values) - np.log(1.0 - neural_values))
    )


def _aligned_probability_arrays(
    neural_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    neural_values = np.asarray(neural_probabilities, dtype=np.float64)
    teacher_values = np.asarray(teacher_probabilities, dtype=np.float64)
    if neural_values.shape != teacher_values.shape:
        raise ValueError(
            "teacher probability shape "
            f"{teacher_values.shape} does not match neural probability shape "
            f"{neural_values.shape}"
        )
    return neural_values, teacher_values


def _finite_values(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def _finite_mean_or_empty(values: np.ndarray) -> float | str:
    finite = _finite_values(values)
    if finite.size == 0:
        return ""
    return float(np.mean(finite))


def _as_basin_embedding_batch(values: np.ndarray) -> np.ndarray:
    embedding = np.asarray(values, dtype=np.float64)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    if embedding.ndim != 2:
        raise ValueError("basin embedding must be a 1D or 2D array")
    if embedding.shape[1] != BASIN_PHASE_EMBEDDING_DIM:
        raise ValueError(
            "basin embedding second dimension must be "
            f"{BASIN_PHASE_EMBEDDING_DIM}; got {embedding.shape[1]}"
        )
    return embedding


def _cosine_similarity_rows(values: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    prototype_values = np.asarray(prototype, dtype=np.float64)
    denominator = (
        np.linalg.norm(values, axis=1) * max(float(np.linalg.norm(prototype_values)), 1e-12)
    )
    return np.divide(
        values @ prototype_values,
        denominator,
        out=np.zeros(values.shape[0], dtype=np.float64),
        where=denominator > 0.0,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def build_basin_phase_representation(
    *,
    actions: np.ndarray,
    payoffs: np.ndarray,
    target_payoff: float,
    action_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Return fixed v1 phase features for prototype basin scoring."""

    action_values = np.asarray(actions, dtype=np.float64)
    payoff_values = np.asarray(payoffs, dtype=np.float64)
    if action_values.ndim == 1:
        action_values = action_values.reshape(1, -1)
    if payoff_values.ndim == 1:
        payoff_values = payoff_values.reshape(1, -1)
    if action_values.shape != payoff_values.shape:
        raise ValueError(
            f"action shape {action_values.shape} must match payoff shape "
            f"{payoff_values.shape}"
        )

    if action_probabilities is None:
        probability_values = action_values
    else:
        probability_values = np.asarray(action_probabilities, dtype=np.float64)
        if probability_values.ndim == 1:
            probability_values = probability_values.reshape(1, -1)
        if probability_values.shape != action_values.shape:
            raise ValueError(
                "action probability shape "
                f"{probability_values.shape} must match action shape "
                f"{action_values.shape}"
            )

    safe_target = max(abs(float(target_payoff)), 1e-8)
    mean_payoff = np.mean(payoff_values, axis=1)
    payoff_alignment = np.clip(
        1.0 - np.abs(mean_payoff - float(target_payoff)) / safe_target,
        0.0,
        1.0,
    )
    payoff_stability = 1.0 / (1.0 + np.std(payoff_values, axis=1) / safe_target)
    action_rate = np.clip(np.mean(action_values, axis=1), 0.0, 1.0)
    policy_rate = np.clip(np.mean(probability_values, axis=1), 0.0, 1.0)
    consensus = np.clip(2.0 * np.abs(action_rate - 0.5), 0.0, 1.0)
    return np.column_stack(
        [
            payoff_alignment,
            action_rate,
            policy_rate,
            consensus,
            payoff_stability,
        ]
    )


def basin_credit_preserves_objective(basin_credit: BasinCreditConfig) -> bool:
    """Return whether enabled basin credit should leave objective advantages intact."""

    return (
        basin_credit.basin_weight == 0.0
        and basin_credit.objective_weight == 0.0
        and basin_credit.individual_weight == 0.0
        and basin_credit.local_social_weight == 0.0
    )


def basin_credit_needs_learned_runtime(basin_credit: BasinCreditConfig) -> bool:
    """Return whether this config needs learned runtime phase-critic scoring."""

    return bool(
        basin_credit.learned_diagnostic_enabled or basin_credit.learned_credit_enabled
    )


def basin_credit_learned_model_path(basin_credit: BasinCreditConfig) -> Path | None:
    """Return the configured learned critic model path for runtime scoring."""

    if basin_credit.learned_credit_enabled and basin_credit.learned_credit_model_path:
        return basin_credit.learned_credit_model_path
    return basin_credit.learned_diagnostic_model_path


def basin_credit_training_candidate_mask(
    *,
    agent_count: int,
    revision_mask: np.ndarray,
    training_scope: BasinCreditTrainingScope,
) -> np.ndarray:
    """Return agents eligible for post-social basin-credit replay training."""

    mask = np.asarray(revision_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("revision_mask must be 1D")
    if mask.size != int(agent_count):
        raise ValueError(
            f"revision_mask length {mask.size} must match agent_count {agent_count}"
        )
    if training_scope == "revised":
        return mask.copy()
    if training_scope == "all":
        return np.ones(int(agent_count), dtype=bool)
    raise ValueError(f"Unsupported basin-credit training_scope: {training_scope}")


def selected_credit_to_action1_advantage(
    selected_action_credit: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Convert observed-vs-ablated credit into action-1-vs-action-0 advantage."""

    credit = np.asarray(selected_action_credit, dtype=np.float64)
    action_values = np.asarray(actions, dtype=np.int64)
    if credit.shape != action_values.shape:
        raise ValueError(
            f"basin credit shape {credit.shape} must match action shape "
            f"{action_values.shape}"
        )
    return np.where(action_values == 1, credit, -credit)


def build_basin_credit_diagnostics(
    *,
    basin_credit: BasinCreditConfig,
    selected_action_credit: np.ndarray,
    score_observed: np.ndarray,
    score_counterfactual: np.ndarray,
    applied_mask: np.ndarray,
    phase_confidence: np.ndarray,
) -> BasinCreditDiagnostics:
    """Return normalized per-agent diagnostics for one-step basin credit."""

    credit = np.asarray(selected_action_credit, dtype=np.float64)
    observed = np.asarray(score_observed, dtype=np.float64)
    counterfactual = np.asarray(score_counterfactual, dtype=np.float64)
    applied = np.asarray(applied_mask, dtype=bool)
    confidence = np.asarray(phase_confidence, dtype=np.float64)
    if not (
        credit.shape
        == observed.shape
        == counterfactual.shape
        == applied.shape
        == confidence.shape
    ):
        raise ValueError("basin-credit diagnostic arrays must have matching shapes")
    return BasinCreditDiagnostics(
        weight=basin_credit.basin_weight,
        objective_weight=basin_credit.objective_weight,
        individual_weight=basin_credit.individual_weight,
        local_social_weight=basin_credit.local_social_weight,
        training_scope=basin_credit.training_scope,
        training_pass_schedule=basin_credit.training_pass_schedule,
        training_passes=basin_credit.training_passes,
        configured_training_passes=basin_credit.training_passes,
        min_training_passes=basin_credit.min_training_passes,
        training_pass_score_threshold=basin_credit.training_pass_score_threshold,
        training_pass_credit_positive_threshold=(
            basin_credit.training_pass_credit_positive_threshold
        ),
        training_pass_credit_delta_threshold=(
            basin_credit.training_pass_credit_delta_threshold
        ),
        target=basin_credit.target_basin,
        selected_action_credit=credit,
        score_observed=observed,
        score_counterfactual=counterfactual,
        applied_mask=applied,
        phase_confidence=confidence,
        critic=basin_credit.critic,
        credit_method=basin_credit.credit_method,
    )


def basin_credit_effective_training_passes(
    *,
    basin_credit: BasinCreditConfig,
    diagnostics: BasinCreditDiagnostics,
) -> int:
    """Return the number of basin replay passes to apply this step."""

    if basin_credit.training_pass_schedule == "fixed":
        return int(basin_credit.training_passes)
    if basin_credit.training_pass_schedule == "target_score_decay":
        applied_scores = diagnostics.score_observed[diagnostics.applied_mask]
        score_mean = _finite_mean_or_empty(applied_scores)
        if (
            isinstance(score_mean, float)
            and score_mean >= basin_credit.training_pass_score_threshold
        ):
            return int(basin_credit.min_training_passes)
        return int(basin_credit.training_passes)
    if basin_credit.training_pass_schedule == "credit_signal_escalation":
        applied_credit = diagnostics.selected_action_credit[diagnostics.applied_mask]
        credit_mean = _finite_mean_or_empty(applied_credit)
        if applied_credit.size == 0 or not isinstance(credit_mean, float):
            return int(basin_credit.min_training_passes)
        positive_rate = float(np.mean(applied_credit > 0.0))
        if (
            credit_mean >= basin_credit.training_pass_credit_delta_threshold
            and positive_rate
            >= basin_credit.training_pass_credit_positive_threshold
        ):
            return int(basin_credit.training_passes)
        return int(basin_credit.min_training_passes)
    raise ValueError(
        "Unsupported basin-credit training_pass_schedule: "
        f"{basin_credit.training_pass_schedule}"
    )


def basin_credit_effective_learned_replay_min_selected_rate(
    *,
    basin_credit: BasinCreditConfig,
    epoch: int,
) -> float:
    """Return the epoch-specific learned replay floor rate."""

    target_rate = float(basin_credit.learned_credit_replay_min_selected_rate)
    if basin_credit.learned_credit_replay_floor_schedule == "fixed":
        return target_rate
    if basin_credit.learned_credit_replay_floor_schedule != "linear_decay":
        raise ValueError(
            "Unsupported learned basin replay floor schedule: "
            f"{basin_credit.learned_credit_replay_floor_schedule}"
        )
    start_rate = float(basin_credit.learned_credit_replay_floor_start_rate)
    decay_epochs = int(basin_credit.learned_credit_replay_floor_decay_epochs)
    if decay_epochs <= 1:
        return target_rate
    progress = min(max((int(epoch) - 1) / float(decay_epochs - 1), 0.0), 1.0)
    return start_rate + (target_rate - start_rate) * progress


def basin_credit_diagnostics_with_training_passes(
    diagnostics: BasinCreditDiagnostics,
    *,
    training_passes: int,
) -> BasinCreditDiagnostics:
    """Return diagnostics annotated with the replay pass count used this step."""

    return replace(diagnostics, training_passes=int(training_passes))


def blend_basin_credit_components(
    components: StateContinuationComponents,
    *,
    diagnostics: BasinCreditDiagnostics,
    basin_credit: BasinCreditConfig,
    actions: np.ndarray,
    basin_action1_advantage: np.ndarray | None = None,
) -> StateContinuationComponents:
    """Blend basin credit with existing objective components for policy learning."""

    if basin_credit_preserves_objective(basin_credit):
        return components

    basin_advantage = (
        selected_credit_to_action1_advantage(
            diagnostics.selected_action_credit,
            actions,
        )
        if basin_action1_advantage is None
        else np.asarray(basin_action1_advantage, dtype=np.float64)
    )
    if basin_advantage.shape != components.effective.shape:
        raise ValueError(
            "basin action1 advantage shape "
            f"{basin_advantage.shape} must match component shape "
            f"{components.effective.shape}"
        )
    effective = (
        basin_credit.objective_weight * components.effective
        + basin_credit.individual_weight * components.material
        + basin_credit.local_social_weight * components.social
        + basin_credit.basin_weight * basin_advantage
    )
    return components.with_effective(effective)


def gradient_cosine(
    base_gradient: np.ndarray,
    distill_gradient: np.ndarray,
) -> float:
    """Return cosine similarity for two flattened gradient vectors."""

    base_values = np.asarray(base_gradient, dtype=np.float64).reshape(-1)
    distill_values = np.asarray(distill_gradient, dtype=np.float64).reshape(-1)
    if base_values.shape != distill_values.shape:
        raise ValueError(
            "base gradient shape "
            f"{base_values.shape} does not match distill gradient shape "
            f"{distill_values.shape}"
        )
    base_norm = float(np.linalg.norm(base_values))
    distill_norm = float(np.linalg.norm(distill_values))
    if base_norm <= 0.0 or distill_norm <= 0.0:
        return float("nan")
    return float(np.dot(base_values, distill_values) / (base_norm * distill_norm))


def objective_teacher_sign_alignment(
    objective_effective: np.ndarray,
    teacher_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return agreement flags and signed objective margin against teacher action."""

    objective_values = np.asarray(objective_effective, dtype=np.float64)
    teacher_values = np.asarray(teacher_probabilities, dtype=np.float64)
    if objective_values.shape != teacher_values.shape:
        raise ValueError(
            "objective effective shape "
            f"{objective_values.shape} does not match teacher probability shape "
            f"{teacher_values.shape}"
        )
    teacher_sign = np.where(teacher_values >= 0.5, 1.0, -1.0)
    objective_sign = np.where(objective_values >= 0.0, 1.0, -1.0)
    margin = teacher_sign * objective_values
    return (teacher_sign == objective_sign).astype(np.float64), margin


def stable_teacher_probability_mask(
    teacher_pre_action: np.ndarray,
    teacher_post_action: np.ndarray,
    *,
    margin_min: float = 0.0,
) -> np.ndarray:
    """Return agents whose teacher action is stable and sufficiently confident."""

    pre_values, post_values = _aligned_probability_arrays(
        teacher_pre_action,
        teacher_post_action,
    )
    if not 0.0 <= margin_min <= 0.5:
        raise ValueError(
            f"teacher stability margin_min must be in [0, 0.5]; got {margin_min:g}"
        )
    pre_actions = pre_values >= 0.5
    post_actions = post_values >= 0.5
    pre_margin = np.abs(pre_values - 0.5)
    post_margin = np.abs(post_values - 0.5)
    return (pre_actions == post_actions) & (pre_margin >= margin_min) & (
        post_margin >= margin_min
    )


def gradient_gate_mask(
    grad_cosines: np.ndarray,
    *,
    min_cosine: float = 0.0,
) -> np.ndarray:
    """Return distill candidates whose gradient cosine is finite and non-conflicting."""

    if not -1.0 <= min_cosine <= 1.0:
        raise ValueError(
            f"gradient gate min_cosine must be in [-1, 1]; got {min_cosine:g}"
        )
    values = np.asarray(grad_cosines, dtype=np.float64)
    return np.isfinite(values) & (values >= min_cosine)


def build_domain_decision_replay_diagnostics(
    *,
    weight: float,
    teacher_probabilities: np.ndarray,
    realized_actions: np.ndarray,
    revision_mask: np.ndarray | None = None,
    stable_teacher_mask: np.ndarray | None = None,
    objective_agreement_mask: np.ndarray | None = None,
    postsocial_improvement_mask: np.ndarray | None = None,
    teacher: str = "reputation_imitation",
) -> DomainDecisionReplayDiagnostics:
    """Build replay gates for teacher-scaffolded decisions."""

    teacher_values = np.asarray(teacher_probabilities, dtype=np.float64)
    if weight <= 0.0 or teacher_values.size == 0:
        empty_float = np.asarray([], dtype=np.float64)
        empty_bool = np.asarray([], dtype=bool)
        empty_int = np.asarray([], dtype=np.int64)
        return DomainDecisionReplayDiagnostics(
            weight=max(float(weight), 0.0),
            teacher_probabilities=empty_float,
            replay_actions=empty_int,
            candidate_mask=empty_bool,
            stable_teacher_mask=empty_bool,
            objective_agreement_mask=empty_bool,
            postsocial_improvement_mask=empty_bool,
            applied_mask=empty_bool,
            teacher=teacher,
        )

    realized_values = np.asarray(realized_actions, dtype=np.int64)
    if realized_values.shape != teacher_values.shape:
        raise ValueError(
            "realized action shape "
            f"{realized_values.shape} does not match teacher probability shape "
            f"{teacher_values.shape}"
        )
    replay_actions = (teacher_values >= 0.5).astype(np.int64)
    candidate = realized_values == replay_actions
    if revision_mask is not None:
        revision_values = np.asarray(revision_mask, dtype=bool)
        if revision_values.shape != teacher_values.shape:
            raise ValueError(
                "revision mask shape "
                f"{revision_values.shape} does not match teacher probability shape "
                f"{teacher_values.shape}"
            )
        candidate = candidate & revision_values

    def resolved_mask(values: np.ndarray | None, name: str) -> np.ndarray:
        if values is None:
            return np.ones_like(candidate, dtype=bool)
        mask_values = np.asarray(values, dtype=bool)
        if mask_values.shape != teacher_values.shape:
            raise ValueError(
                f"{name} shape {mask_values.shape} does not match teacher "
                f"probability shape {teacher_values.shape}"
            )
        return mask_values

    stable = resolved_mask(stable_teacher_mask, "stable teacher mask")
    objective = resolved_mask(objective_agreement_mask, "objective agreement mask")
    postsocial = resolved_mask(
        postsocial_improvement_mask,
        "postsocial improvement mask",
    )
    applied = candidate & stable & objective & postsocial
    return DomainDecisionReplayDiagnostics(
        weight=float(weight),
        teacher_probabilities=teacher_values,
        replay_actions=replay_actions,
        candidate_mask=candidate,
        stable_teacher_mask=stable,
        objective_agreement_mask=objective,
        postsocial_improvement_mask=postsocial,
        applied_mask=applied,
        teacher=teacher,
    )


def build_domain_teacher_alignment_diagnostics(
    *,
    teacher_pre_action: np.ndarray,
    teacher_post_action: np.ndarray,
    pre_local: np.ndarray,
    post_local: np.ndarray,
    post_social: np.ndarray,
    realized_actions: np.ndarray,
    objective_effective: np.ndarray | None = None,
    base_losses: np.ndarray | None = None,
    distill_losses: np.ndarray | None = None,
    base_grad_norms: np.ndarray | None = None,
    distill_grad_norms: np.ndarray | None = None,
    grad_cosines: np.ndarray | None = None,
    distill_candidate_mask: np.ndarray | None = None,
    distill_stable_teacher_mask: np.ndarray | None = None,
    distill_gradient_gate_mask: np.ndarray | None = None,
    distill_applied_mask: np.ndarray | None = None,
    teacher: str = "reputation_imitation",
) -> DomainTeacherAlignmentDiagnostics | None:
    """Build teacher-alignment diagnostics when full-path arrays are available."""

    teacher_pre = np.asarray(teacher_pre_action, dtype=np.float64)
    if teacher_pre.size == 0:
        return None
    teacher_post = np.asarray(teacher_post_action, dtype=np.float64)
    pre_values = np.asarray(pre_local, dtype=np.float64)
    post_local_values = np.asarray(post_local, dtype=np.float64)
    post_social_values = np.asarray(post_social, dtype=np.float64)
    realized_values = np.asarray(realized_actions, dtype=np.int64)
    for name, values in (
        ("teacher_post_action", teacher_post),
        ("pre_local", pre_values),
        ("post_local", post_local_values),
        ("post_social", post_social_values),
        ("realized_actions", realized_values),
    ):
        if values.shape != teacher_pre.shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match teacher_pre_action "
                f"shape {teacher_pre.shape}"
            )

    empty = np.asarray([], dtype=np.float64)
    objective_values = (
        empty
        if objective_effective is None
        else np.asarray(objective_effective, dtype=np.float64)
    )
    if objective_values.size > 0 and objective_values.shape != teacher_pre.shape:
        raise ValueError(
            "objective_effective shape "
            f"{objective_values.shape} does not match teacher_pre_action shape "
            f"{teacher_pre.shape}"
        )

    def optional(values: np.ndarray | None) -> np.ndarray:
        return empty if values is None else np.asarray(values, dtype=np.float64)

    def optional_mask(values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.asarray([], dtype=bool)
        mask_values = np.asarray(values, dtype=bool)
        if mask_values.size == 0:
            return mask_values
        if mask_values.shape != teacher_pre.shape:
            raise ValueError(
                "distill mask shape "
                f"{mask_values.shape} does not match teacher_pre_action shape "
                f"{teacher_pre.shape}"
            )
        return mask_values

    return DomainTeacherAlignmentDiagnostics(
        teacher=teacher,
        teacher_pre_action=teacher_pre,
        teacher_post_action=teacher_post,
        pre_local=pre_values,
        post_local=post_local_values,
        post_social=post_social_values,
        realized_actions=realized_values,
        objective_effective=objective_values,
        base_losses=optional(base_losses),
        distill_losses=optional(distill_losses),
        base_grad_norms=optional(base_grad_norms),
        distill_grad_norms=optional(distill_grad_norms),
        grad_cosines=optional(grad_cosines),
        distill_candidate_mask=optional_mask(distill_candidate_mask),
        distill_stable_teacher_mask=optional_mask(distill_stable_teacher_mask),
        distill_gradient_gate_mask=optional_mask(distill_gradient_gate_mask),
        distill_applied_mask=optional_mask(distill_applied_mask),
    )


def blend_domain_bootstrap_components(
    components: StateContinuationComponents,
    *,
    teacher_probabilities: np.ndarray,
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> tuple[StateContinuationComponents, DomainBootstrapDiagnostics | None]:
    """Return training components with bootstrap-blended effective advantages."""

    if not bootstrap.enabled:
        return components, None
    weight = domain_bootstrap_weight(bootstrap, epoch)
    if weight <= 0.0:
        return components, DomainBootstrapDiagnostics(
            weight=0.0,
            teacher_signed=np.asarray([], dtype=np.float64),
            bootstrapped_effective=np.asarray([], dtype=np.float64),
            teacher=bootstrap.teacher,
        )
    teacher_signed = teacher_probabilities_to_signed_advantages(
        teacher_probabilities,
        teacher_scale=bootstrap.teacher_scale,
    )
    bootstrapped_effective = blend_bootstrap_signed_advantages(
        components.effective,
        teacher_signed,
        weight=weight,
    )
    return (
        components.with_effective(bootstrapped_effective),
        DomainBootstrapDiagnostics(
            weight=weight,
            teacher_signed=teacher_signed,
            bootstrapped_effective=bootstrapped_effective,
            teacher=bootstrap.teacher,
        ),
    )


def blend_domain_decision_bootstrap_probabilities(
    neural_probabilities: np.ndarray,
    *,
    teacher_probabilities: np.ndarray,
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> tuple[np.ndarray, DomainDecisionBootstrapDiagnostics | None]:
    """Return decision probabilities blended with the scheduled teacher signal."""

    neural_values = np.asarray(neural_probabilities, dtype=np.float64)
    if not bootstrap.decision_enabled:
        return neural_values, None
    weight = domain_decision_bootstrap_weight(bootstrap, epoch)
    if weight <= 0.0:
        return neural_values, DomainDecisionBootstrapDiagnostics(
            weight=0.0,
            teacher_probabilities=np.asarray([], dtype=np.float64),
            neural_probabilities=np.asarray([], dtype=np.float64),
            bootstrapped_probabilities=np.asarray([], dtype=np.float64),
            teacher=bootstrap.teacher,
        )
    bootstrapped_probabilities = blend_bootstrap_decision_probabilities(
        neural_values,
        teacher_probabilities,
        weight=weight,
    )
    return (
        bootstrapped_probabilities,
        DomainDecisionBootstrapDiagnostics(
            weight=weight,
            teacher_probabilities=np.asarray(teacher_probabilities, dtype=np.float64),
            neural_probabilities=neural_values,
            bootstrapped_probabilities=bootstrapped_probabilities,
            teacher=bootstrap.teacher,
        ),
    )


def domain_distill_bootstrap_diagnostic_components(
    *,
    neural_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
    realized_actions: np.ndarray,
    bootstrap: DomainBootstrapConfig,
    epoch: int,
) -> DomainDistillBootstrapDiagnostics | None:
    """Return diagnostics for teacher policy distillation at this epoch."""

    if not bootstrap.distill_enabled:
        return None
    weight = domain_distill_bootstrap_weight(bootstrap, epoch)
    if weight <= 0.0:
        return DomainDistillBootstrapDiagnostics(
            weight=0.0,
            teacher_probabilities=np.asarray([], dtype=np.float64),
            neural_probabilities=np.asarray([], dtype=np.float64),
            bce=np.asarray([], dtype=np.float64),
            kl=np.asarray([], dtype=np.float64),
            argmax_agreement=np.asarray([], dtype=np.float64),
            realized_action_agreement=np.asarray([], dtype=np.float64),
            teacher=bootstrap.distill_teacher,
        )
    neural_values, teacher_values = _aligned_probability_arrays(
        neural_probabilities,
        teacher_probabilities,
    )
    realized_values = np.asarray(realized_actions, dtype=np.int64)
    if realized_values.shape != teacher_values.shape:
        raise ValueError(
            "realized action shape "
            f"{realized_values.shape} does not match teacher probability shape "
            f"{teacher_values.shape}"
        )
    teacher_actions = (teacher_values >= 0.5).astype(np.int64)
    neural_actions = (neural_values >= 0.5).astype(np.int64)
    return DomainDistillBootstrapDiagnostics(
        weight=weight,
        teacher_probabilities=teacher_values,
        neural_probabilities=neural_values,
        bce=teacher_policy_bce(neural_values, teacher_values),
        kl=teacher_policy_kl(neural_values, teacher_values),
        argmax_agreement=(teacher_actions == neural_actions).astype(np.float64),
        realized_action_agreement=(teacher_actions == realized_values).astype(
            np.float64
        ),
        teacher=bootstrap.distill_teacher,
    )


def state_continuation_diagnostics(
    components: StateContinuationComponents | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for objective component arrays."""

    if components is None or components.effective.size == 0:
        return {
            "domain_material_advantage_mean": "",
            "domain_social_continuation_advantage_mean": "",
            "domain_welfare_advantage_mean": "",
            "domain_environment_continuation_advantage_mean": "",
            "domain_risk_advantage_mean": "",
            "domain_linear_advantage_mean": "",
            "domain_interaction_advantage_mean": "",
            "domain_activation_input_mean": "",
            "domain_effective_advantage_mean": "",
            "domain_effective_advantage_positive_rate": "",
            "domain_objective_profile": "",
        }
    return {
        "domain_material_advantage_mean": float(np.mean(components.material)),
        "domain_social_continuation_advantage_mean": float(np.mean(components.social)),
        "domain_welfare_advantage_mean": float(np.mean(components.welfare)),
        "domain_environment_continuation_advantage_mean": float(
            np.mean(components.environment)
        ),
        "domain_risk_advantage_mean": float(np.mean(components.risk)),
        "domain_linear_advantage_mean": float(np.mean(components.linear)),
        "domain_interaction_advantage_mean": float(np.mean(components.interaction)),
        "domain_activation_input_mean": float(np.mean(components.activation_input)),
        "domain_effective_advantage_mean": float(np.mean(components.effective)),
        "domain_effective_advantage_positive_rate": float(
            np.mean(components.effective >= 0.0)
        ),
        "domain_objective_profile": components.objective_profile,
    }


def domain_bootstrap_diagnostics(
    diagnostics: DomainBootstrapDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for domain-bootstrap teacher blending."""

    if diagnostics is None:
        return {
            "domain_bootstrap_weight": "",
            "domain_teacher_signed_advantage_mean": "",
            "domain_bootstrapped_effective_advantage_mean": "",
            "domain_bootstrap_teacher": "",
        }
    if diagnostics.teacher_signed.size == 0:
        return {
            "domain_bootstrap_weight": diagnostics.weight,
            "domain_teacher_signed_advantage_mean": "",
            "domain_bootstrapped_effective_advantage_mean": "",
            "domain_bootstrap_teacher": diagnostics.teacher,
        }
    return {
        "domain_bootstrap_weight": diagnostics.weight,
        "domain_teacher_signed_advantage_mean": float(
            np.mean(diagnostics.teacher_signed)
        ),
        "domain_bootstrapped_effective_advantage_mean": float(
            np.mean(diagnostics.bootstrapped_effective)
        ),
        "domain_bootstrap_teacher": diagnostics.teacher,
    }


def domain_decision_bootstrap_diagnostics(
    diagnostics: DomainDecisionBootstrapDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for decision-bootstrap probability blending."""

    if diagnostics is None:
        return {
            "domain_decision_bootstrap_weight": "",
            "domain_teacher_decision_probability_mean": "",
            "domain_neural_decision_probability_mean": "",
            "domain_bootstrapped_decision_probability_mean": "",
            "domain_decision_bootstrap_teacher": "",
        }
    if diagnostics.teacher_probabilities.size == 0:
        return {
            "domain_decision_bootstrap_weight": diagnostics.weight,
            "domain_teacher_decision_probability_mean": "",
            "domain_neural_decision_probability_mean": "",
            "domain_bootstrapped_decision_probability_mean": "",
            "domain_decision_bootstrap_teacher": "",
        }
    return {
        "domain_decision_bootstrap_weight": diagnostics.weight,
        "domain_teacher_decision_probability_mean": float(
            np.mean(diagnostics.teacher_probabilities)
        ),
        "domain_neural_decision_probability_mean": float(
            np.mean(diagnostics.neural_probabilities)
        ),
        "domain_bootstrapped_decision_probability_mean": float(
            np.mean(diagnostics.bootstrapped_probabilities)
        ),
        "domain_decision_bootstrap_teacher": diagnostics.teacher,
    }


def domain_decision_replay_diagnostics(
    diagnostics: DomainDecisionReplayDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for teacher-scaffold decision replay."""

    if diagnostics is None:
        return dict(_DOMAIN_DECISION_REPLAY_EMPTY_FIELDS)
    if diagnostics.candidate_mask.size == 0:
        fields = dict(_DOMAIN_DECISION_REPLAY_EMPTY_FIELDS)
        fields["domain_decision_replay_weight"] = diagnostics.weight
        return fields

    candidate = diagnostics.candidate_mask
    stable = diagnostics.stable_teacher_mask
    objective = diagnostics.objective_agreement_mask
    postsocial = diagnostics.postsocial_improvement_mask
    return {
        "domain_decision_replay_weight": diagnostics.weight,
        "domain_decision_replay_candidate_rate": float(np.mean(candidate)),
        "domain_decision_replay_applied_rate": float(
            np.mean(diagnostics.applied_mask)
        ),
        "domain_decision_replay_rejected_unstable_teacher_rate": float(
            np.mean(candidate & ~stable)
        ),
        "domain_decision_replay_rejected_objective_rate": float(
            np.mean(candidate & stable & ~objective)
        ),
        "domain_decision_replay_rejected_postsocial_rate": float(
            np.mean(candidate & stable & objective & ~postsocial)
        ),
        "domain_decision_replay_teacher_probability_mean": float(
            np.mean(diagnostics.teacher_probabilities)
        ),
        "domain_decision_replay_teacher": diagnostics.teacher,
    }


def basin_credit_diagnostics(
    diagnostics: BasinCreditDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for counterfactual basin credit."""

    if diagnostics is None:
        return dict(_DOMAIN_BASIN_CREDIT_EMPTY_FIELDS)
    applied = diagnostics.applied_mask
    applied_credit = diagnostics.selected_action_credit[applied]
    applied_confidence = diagnostics.phase_confidence[applied]
    candidate_rate: float | str = "" if applied.size == 0 else float(np.mean(applied))
    if applied_credit.size == 0:
        return {
            "domain_basin_training_scope": diagnostics.training_scope,
            "domain_basin_training_pass_schedule": (
                diagnostics.training_pass_schedule
            ),
            "domain_basin_training_passes": diagnostics.training_passes,
            "domain_basin_training_passes_configured": (
                diagnostics.configured_training_passes
            ),
            "domain_basin_min_training_passes": diagnostics.min_training_passes,
            "domain_basin_training_pass_score_threshold": (
                diagnostics.training_pass_score_threshold
            ),
            "domain_basin_training_pass_credit_positive_threshold": (
                diagnostics.training_pass_credit_positive_threshold
            ),
            "domain_basin_training_pass_credit_delta_threshold": (
                diagnostics.training_pass_credit_delta_threshold
            ),
            "domain_basin_training_candidate_rate": candidate_rate,
            "domain_basin_objective_weight": diagnostics.objective_weight,
            "domain_basin_individual_weight": diagnostics.individual_weight,
            "domain_basin_local_social_weight": diagnostics.local_social_weight,
            "domain_basin_credit_weight": diagnostics.weight,
            "domain_basin_score_mean": _finite_mean_or_empty(
                diagnostics.score_observed
            ),
            "domain_basin_score_delta_mean": "",
            "domain_basin_credit_positive_rate": "",
            "domain_basin_phase_confidence_mean": _finite_mean_or_empty(
                diagnostics.phase_confidence
            ),
            "domain_basin_target": diagnostics.target,
        }
    return {
        "domain_basin_training_scope": diagnostics.training_scope,
        "domain_basin_training_pass_schedule": diagnostics.training_pass_schedule,
        "domain_basin_training_passes": diagnostics.training_passes,
        "domain_basin_training_passes_configured": (
            diagnostics.configured_training_passes
        ),
        "domain_basin_min_training_passes": diagnostics.min_training_passes,
        "domain_basin_training_pass_score_threshold": (
            diagnostics.training_pass_score_threshold
        ),
        "domain_basin_training_pass_credit_positive_threshold": (
            diagnostics.training_pass_credit_positive_threshold
        ),
        "domain_basin_training_pass_credit_delta_threshold": (
            diagnostics.training_pass_credit_delta_threshold
        ),
        "domain_basin_training_candidate_rate": candidate_rate,
        "domain_basin_objective_weight": diagnostics.objective_weight,
        "domain_basin_individual_weight": diagnostics.individual_weight,
        "domain_basin_local_social_weight": diagnostics.local_social_weight,
        "domain_basin_credit_weight": diagnostics.weight,
        "domain_basin_score_mean": _finite_mean_or_empty(diagnostics.score_observed),
        "domain_basin_score_delta_mean": float(np.mean(applied_credit)),
        "domain_basin_credit_positive_rate": float(np.mean(applied_credit > 0.0)),
        "domain_basin_phase_confidence_mean": _finite_mean_or_empty(
            applied_confidence
        ),
        "domain_basin_target": diagnostics.target,
    }


def basin_credit_training_diagnostics(
    *,
    diagnostics: BasinCreditDiagnostics | None,
    training_components: StateContinuationComponents | None,
    actions: np.ndarray | None,
    training_action1_advantage: np.ndarray | None = None,
    training_credit_source: str = "prototype",
    training_replay_selection: str = "all",
    training_replay_min_selected_rate: float | str = "",
    training_replay_mask: np.ndarray | None = None,
    training_replay_weight: np.ndarray | None = None,
    learned_credit_used_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Return aggregate diagnostics for the actual basin-credit training signal."""

    if diagnostics is None or training_components is None or actions is None:
        return dict(_DOMAIN_BASIN_CREDIT_TRAINING_EMPTY_FIELDS)
    action1_advantage = (
        selected_credit_to_action1_advantage(
            diagnostics.selected_action_credit,
            actions,
        )
        if training_action1_advantage is None
        else np.asarray(training_action1_advantage, dtype=np.float64)
    )
    training_effective = np.asarray(training_components.effective, dtype=np.float64)
    if action1_advantage.size == 0 or training_effective.size == 0:
        return dict(_DOMAIN_BASIN_CREDIT_TRAINING_EMPTY_FIELDS)
    replay_mask = (
        np.ones(action1_advantage.shape, dtype=bool)
        if training_replay_mask is None
        else np.asarray(training_replay_mask, dtype=bool)
    )
    if replay_mask.shape != action1_advantage.shape:
        raise ValueError(
            "training replay mask shape "
            f"{replay_mask.shape} must match action1 advantage shape "
            f"{action1_advantage.shape}"
        )
    replay_weight = (
        replay_mask.astype(np.float64)
        if training_replay_weight is None
        else np.asarray(training_replay_weight, dtype=np.float64)
    )
    if replay_weight.shape != action1_advantage.shape:
        raise ValueError(
            "training replay weight shape "
            f"{replay_weight.shape} must match action1 advantage shape "
            f"{action1_advantage.shape}"
        )
    replay_weight = np.where(replay_mask, np.clip(replay_weight, 0.0, 1.0), 0.0)
    learned_rate: float | str = ""
    if learned_credit_used_mask is not None:
        learned_mask = np.asarray(learned_credit_used_mask, dtype=bool)
        if learned_mask.shape != action1_advantage.shape:
            raise ValueError(
                "learned credit mask shape "
                f"{learned_mask.shape} must match action1 advantage shape "
                f"{action1_advantage.shape}"
            )
        learned_rate = float(np.mean(learned_mask))
    selected_mask = replay_mask & (replay_weight > 0.0)
    selected_action1 = action1_advantage[selected_mask]
    selected_effective = training_effective[selected_mask]
    selected_weights = replay_weight[selected_mask]
    if selected_action1.size == 0 or selected_effective.size == 0:
        return {
            "domain_basin_training_credit_source": training_credit_source,
            "domain_basin_training_replay_selection": training_replay_selection,
            "domain_basin_training_replay_min_selected_rate": (
                training_replay_min_selected_rate
            ),
            "domain_basin_training_replay_selected_rate": float(np.mean(selected_mask)),
            "domain_basin_training_replay_weight_mean": float(np.mean(replay_weight)),
            "domain_basin_training_replay_weight_positive_rate": float(
                np.mean(replay_weight > 0.0)
            ),
            "domain_basin_training_learned_credit_rate": learned_rate,
            "domain_basin_action1_advantage_mean": "",
            "domain_basin_action1_advantage_positive_rate": "",
            "domain_basin_training_effective_advantage_mean": "",
            "domain_basin_training_effective_advantage_positive_rate": "",
            "domain_basin_training_effective_advantage_abs_mean": "",
        }
    return {
        "domain_basin_training_credit_source": training_credit_source,
        "domain_basin_training_replay_selection": training_replay_selection,
        "domain_basin_training_replay_min_selected_rate": (
            training_replay_min_selected_rate
        ),
        "domain_basin_training_replay_selected_rate": float(np.mean(selected_mask)),
        "domain_basin_training_replay_weight_mean": float(np.mean(replay_weight)),
        "domain_basin_training_replay_weight_positive_rate": float(
            np.mean(replay_weight > 0.0)
        ),
        "domain_basin_training_learned_credit_rate": learned_rate,
        "domain_basin_action1_advantage_mean": float(
            np.average(selected_action1, weights=selected_weights)
        ),
        "domain_basin_action1_advantage_positive_rate": float(
            np.average(selected_action1 > 0.0, weights=selected_weights)
        ),
        "domain_basin_training_effective_advantage_mean": float(
            np.average(selected_effective, weights=selected_weights)
        ),
        "domain_basin_training_effective_advantage_positive_rate": float(
            np.average(selected_effective > 0.0, weights=selected_weights)
        ),
        "domain_basin_training_effective_advantage_abs_mean": float(
            np.average(np.abs(selected_effective), weights=selected_weights)
        ),
    }


def domain_distill_bootstrap_diagnostics(
    diagnostics: DomainDistillBootstrapDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for teacher policy distillation."""

    if diagnostics is None:
        return {
            "domain_distill_bootstrap_weight": "",
            "domain_distill_teacher": "",
            "domain_teacher_policy_bce_mean": "",
            "domain_teacher_policy_kl_mean": "",
            "domain_teacher_neural_probability_mean": "",
            "domain_teacher_probability_mean": "",
            "domain_teacher_neural_argmax_agreement": "",
            "domain_teacher_realized_action_agreement": "",
        }
    if diagnostics.teacher_probabilities.size == 0:
        return {
            "domain_distill_bootstrap_weight": diagnostics.weight,
            "domain_distill_teacher": "",
            "domain_teacher_policy_bce_mean": "",
            "domain_teacher_policy_kl_mean": "",
            "domain_teacher_neural_probability_mean": "",
            "domain_teacher_probability_mean": "",
            "domain_teacher_neural_argmax_agreement": "",
            "domain_teacher_realized_action_agreement": "",
        }
    return {
        "domain_distill_bootstrap_weight": diagnostics.weight,
        "domain_distill_teacher": diagnostics.teacher,
        "domain_teacher_policy_bce_mean": float(np.mean(diagnostics.bce)),
        "domain_teacher_policy_kl_mean": float(np.mean(diagnostics.kl)),
        "domain_teacher_neural_probability_mean": float(
            np.mean(diagnostics.neural_probabilities)
        ),
        "domain_teacher_probability_mean": float(
            np.mean(diagnostics.teacher_probabilities)
        ),
        "domain_teacher_neural_argmax_agreement": float(
            np.mean(diagnostics.argmax_agreement)
        ),
        "domain_teacher_realized_action_agreement": float(
            np.mean(diagnostics.realized_action_agreement)
        ),
    }


def domain_teacher_alignment_diagnostics(
    diagnostics: DomainTeacherAlignmentDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate diagnostics for teacher/neural alignment flow."""

    if diagnostics is None or diagnostics.teacher_pre_action.size == 0:
        return dict(_DOMAIN_TEACHER_ALIGNMENT_EMPTY_FIELDS)
    teacher_pre = diagnostics.teacher_pre_action
    bce_pre = teacher_policy_bce(diagnostics.pre_local, teacher_pre)
    bce_post_local = teacher_policy_bce(diagnostics.post_local, teacher_pre)
    bce_post_social = teacher_policy_bce(diagnostics.post_social, teacher_pre)
    kl_pre = teacher_policy_kl(diagnostics.pre_local, teacher_pre)
    kl_post_local = teacher_policy_kl(diagnostics.post_local, teacher_pre)
    kl_post_social = teacher_policy_kl(diagnostics.post_social, teacher_pre)

    teacher_actions = (teacher_pre >= 0.5).astype(np.int64)
    pre_actions = (diagnostics.pre_local >= 0.5).astype(np.int64)
    post_local_actions = (diagnostics.post_local >= 0.5).astype(np.int64)
    post_social_actions = (diagnostics.post_social >= 0.5).astype(np.int64)
    objective_agreement: np.ndarray = np.asarray([], dtype=np.float64)
    objective_margin: np.ndarray = np.asarray([], dtype=np.float64)
    if diagnostics.objective_effective.size > 0:
        objective_agreement, objective_margin = objective_teacher_sign_alignment(
            diagnostics.objective_effective,
            teacher_pre,
        )
    finite_cosines = _finite_values(diagnostics.grad_cosines)
    candidate_mask = diagnostics.distill_candidate_mask
    applied_mask = diagnostics.distill_applied_mask
    stable_mask = diagnostics.distill_stable_teacher_mask
    gradient_mask = diagnostics.distill_gradient_gate_mask
    candidate_rate: float | str = ""
    applied_rate: float | str = ""
    rejected_unstable_rate: float | str = ""
    rejected_gradient_rate: float | str = ""
    if candidate_mask.size > 0:
        candidate_rate = float(np.mean(candidate_mask))
        applied_rate = float(np.mean(applied_mask)) if applied_mask.size > 0 else 0.0
        if stable_mask.size > 0:
            rejected_unstable_rate = float(np.mean(candidate_mask & ~stable_mask))
        if gradient_mask.size > 0 and stable_mask.size > 0:
            rejected_gradient_rate = float(
                np.mean(candidate_mask & stable_mask & ~gradient_mask)
            )
    return {
        "domain_teacher_policy_bce_pre_local_mean": float(np.mean(bce_pre)),
        "domain_teacher_policy_bce_post_local_mean": float(np.mean(bce_post_local)),
        "domain_teacher_policy_bce_post_social_mean": float(np.mean(bce_post_social)),
        "domain_teacher_policy_kl_pre_local_mean": float(np.mean(kl_pre)),
        "domain_teacher_policy_kl_post_local_mean": float(np.mean(kl_post_local)),
        "domain_teacher_policy_kl_post_social_mean": float(np.mean(kl_post_social)),
        "domain_teacher_bce_delta_local": float(np.mean(bce_post_local - bce_pre)),
        "domain_teacher_bce_delta_social": float(
            np.mean(bce_post_social - bce_post_local)
        ),
        "domain_teacher_neural_argmax_agreement_pre_local": float(
            np.mean(pre_actions == teacher_actions)
        ),
        "domain_teacher_neural_argmax_agreement_post_local": float(
            np.mean(post_local_actions == teacher_actions)
        ),
        "domain_teacher_neural_argmax_agreement_post_social": float(
            np.mean(post_social_actions == teacher_actions)
        ),
        "domain_teacher_realized_action_agreement": float(
            np.mean(diagnostics.realized_actions == teacher_actions)
        ),
        "domain_teacher_probability_pre_action_mean": float(np.mean(teacher_pre)),
        "domain_teacher_probability_post_action_mean": float(
            np.mean(diagnostics.teacher_post_action)
        ),
        "domain_teacher_target_shift_mean": float(
            np.mean(np.abs(diagnostics.teacher_post_action - teacher_pre))
        ),
        "domain_teacher_target_flip_rate": float(
            np.mean((diagnostics.teacher_post_action >= 0.5) != (teacher_pre >= 0.5))
        ),
        "domain_effective_advantage_teacher_sign_agreement": _finite_mean_or_empty(
            objective_agreement
        ),
        "domain_effective_advantage_teacher_sign_conflict_rate": (
            ""
            if objective_agreement.size == 0
            else float(np.mean(1.0 - objective_agreement))
        ),
        "domain_effective_advantage_teacher_margin_mean": _finite_mean_or_empty(
            objective_margin
        ),
        "domain_base_loss_mean": _finite_mean_or_empty(diagnostics.base_losses),
        "domain_distill_loss_mean": _finite_mean_or_empty(diagnostics.distill_losses),
        "domain_base_grad_norm_mean": _finite_mean_or_empty(
            diagnostics.base_grad_norms
        ),
        "domain_distill_grad_norm_mean": _finite_mean_or_empty(
            diagnostics.distill_grad_norms
        ),
        "domain_base_distill_grad_cosine_mean": _finite_mean_or_empty(
            diagnostics.grad_cosines
        ),
        "domain_base_distill_grad_cosine_negative_rate": (
            "" if finite_cosines.size == 0 else float(np.mean(finite_cosines < 0.0))
        ),
        "domain_distill_candidate_rate": candidate_rate,
        "domain_distill_applied_rate": applied_rate,
        "domain_distill_rejected_unstable_teacher_rate": rejected_unstable_rate,
        "domain_distill_rejected_gradient_rate": rejected_gradient_rate,
        "domain_teacher_alignment_teacher": diagnostics.teacher,
    }


def state_continuation_micro_fields(
    components: StateContinuationComponents | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent objective component diagnostics."""

    if components is None or agent_id >= components.effective.size:
        return {
            "domain_material_advantage": "",
            "domain_social_continuation_advantage": "",
            "domain_welfare_advantage": "",
            "domain_environment_continuation_advantage": "",
            "domain_risk_advantage": "",
            "domain_linear_advantage": "",
            "domain_interaction_advantage": "",
            "domain_activation_input": "",
            "domain_effective_advantage": "",
            "domain_objective_profile": "",
        }
    return {
        "domain_material_advantage": float(components.material[agent_id]),
        "domain_social_continuation_advantage": float(components.social[agent_id]),
        "domain_welfare_advantage": float(components.welfare[agent_id]),
        "domain_environment_continuation_advantage": float(
            components.environment[agent_id]
        ),
        "domain_risk_advantage": float(components.risk[agent_id]),
        "domain_linear_advantage": float(components.linear[agent_id]),
        "domain_interaction_advantage": float(components.interaction[agent_id]),
        "domain_activation_input": float(components.activation_input[agent_id]),
        "domain_effective_advantage": float(components.effective[agent_id]),
        "domain_objective_profile": components.objective_profile,
    }


def domain_bootstrap_micro_fields(
    diagnostics: DomainBootstrapDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent bootstrap diagnostics using the aggregate field contract."""

    if diagnostics is None:
        return {
            "domain_bootstrap_weight": "",
            "domain_teacher_signed_advantage_mean": "",
            "domain_bootstrapped_effective_advantage_mean": "",
            "domain_bootstrap_teacher": "",
        }
    if agent_id >= diagnostics.teacher_signed.size:
        return {
            "domain_bootstrap_weight": diagnostics.weight,
            "domain_teacher_signed_advantage_mean": "",
            "domain_bootstrapped_effective_advantage_mean": "",
            "domain_bootstrap_teacher": diagnostics.teacher,
        }
    return {
        "domain_bootstrap_weight": diagnostics.weight,
        "domain_teacher_signed_advantage_mean": float(
            diagnostics.teacher_signed[agent_id]
        ),
        "domain_bootstrapped_effective_advantage_mean": float(
            diagnostics.bootstrapped_effective[agent_id]
        ),
        "domain_bootstrap_teacher": diagnostics.teacher,
    }


def basin_credit_micro_fields(
    diagnostics: BasinCreditDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent basin-credit diagnostics."""

    if diagnostics is None or agent_id >= diagnostics.applied_mask.size:
        return dict(_DOMAIN_BASIN_CREDIT_MICRO_EMPTY_FIELDS)
    applied = bool(diagnostics.applied_mask[agent_id])
    fields = {
        "domain_basin_credit_applied": applied,
        "domain_basin_score_observed": float(diagnostics.score_observed[agent_id]),
    }
    if not applied:
        return {
            **fields,
            "domain_basin_credit": "",
            "domain_basin_score_counterfactual": "",
        }
    return {
        **fields,
        "domain_basin_credit": float(diagnostics.selected_action_credit[agent_id]),
        "domain_basin_score_counterfactual": float(
            diagnostics.score_counterfactual[agent_id]
        ),
    }


def basin_credit_training_micro_fields(
    *,
    diagnostics: BasinCreditDiagnostics | None,
    training_components: StateContinuationComponents | None,
    actions: np.ndarray | None,
    agent_id: int,
    training_action1_advantage: np.ndarray | None = None,
    training_credit_source: str = "prototype",
    training_replay_selection: str = "all",
    training_replay_mask: np.ndarray | None = None,
    training_replay_weight: np.ndarray | None = None,
    learned_credit_used_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Return per-agent diagnostics for the basin-credit training signal."""

    if (
        diagnostics is None
        or training_components is None
        or actions is None
        or agent_id >= diagnostics.selected_action_credit.size
        or agent_id >= training_components.effective.size
    ):
        return dict(_DOMAIN_BASIN_CREDIT_TRAINING_MICRO_EMPTY_FIELDS)
    action1_advantage = (
        selected_credit_to_action1_advantage(
            diagnostics.selected_action_credit,
            actions,
        )
        if training_action1_advantage is None
        else np.asarray(training_action1_advantage, dtype=np.float64)
    )
    if action1_advantage.shape != diagnostics.selected_action_credit.shape:
        return dict(_DOMAIN_BASIN_CREDIT_TRAINING_MICRO_EMPTY_FIELDS)
    if training_replay_mask is None:
        replay_selected = True
    else:
        replay_values = np.asarray(training_replay_mask, dtype=bool)
        replay_selected = (
            bool(replay_values[agent_id])
            if replay_values.shape == diagnostics.selected_action_credit.shape
            else False
        )
    replay_weight: float | str
    if training_replay_weight is None:
        replay_weight = 1.0 if replay_selected else 0.0
    else:
        replay_weight_values = np.asarray(training_replay_weight, dtype=np.float64)
        replay_weight = (
            float(np.clip(replay_weight_values[agent_id], 0.0, 1.0))
            if replay_weight_values.shape == diagnostics.selected_action_credit.shape
            else 0.0
        )
        replay_selected = replay_selected and replay_weight > 0.0
    learned_credit_used: bool | str = ""
    if learned_credit_used_mask is not None:
        learned_values = np.asarray(learned_credit_used_mask, dtype=bool)
        if learned_values.shape == diagnostics.selected_action_credit.shape:
            learned_credit_used = bool(learned_values[agent_id])
    if not replay_selected:
        return {
            "domain_basin_training_credit_source": training_credit_source,
            "domain_basin_training_replay_selection": training_replay_selection,
            "domain_basin_training_replay_selected": False,
            "domain_basin_training_replay_weight": replay_weight,
            "domain_basin_training_learned_credit_used": learned_credit_used,
            "domain_basin_action1_advantage": "",
            "domain_basin_training_effective_advantage": "",
        }
    return {
        "domain_basin_training_credit_source": training_credit_source,
        "domain_basin_training_replay_selection": training_replay_selection,
        "domain_basin_training_replay_selected": True,
        "domain_basin_training_replay_weight": replay_weight,
        "domain_basin_training_learned_credit_used": learned_credit_used,
        "domain_basin_action1_advantage": float(action1_advantage[agent_id]),
        "domain_basin_training_effective_advantage": float(
            training_components.effective[agent_id]
        ),
    }


def domain_decision_bootstrap_micro_fields(
    diagnostics: DomainDecisionBootstrapDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent decision-bootstrap diagnostics with aggregate field names."""

    if diagnostics is None:
        return {
            "domain_decision_bootstrap_weight": "",
            "domain_teacher_decision_probability_mean": "",
            "domain_neural_decision_probability_mean": "",
            "domain_bootstrapped_decision_probability_mean": "",
            "domain_decision_bootstrap_teacher": "",
        }
    if agent_id >= diagnostics.teacher_probabilities.size:
        return {
            "domain_decision_bootstrap_weight": diagnostics.weight,
            "domain_teacher_decision_probability_mean": "",
            "domain_neural_decision_probability_mean": "",
            "domain_bootstrapped_decision_probability_mean": "",
            "domain_decision_bootstrap_teacher": "",
        }
    return {
        "domain_decision_bootstrap_weight": diagnostics.weight,
        "domain_teacher_decision_probability_mean": float(
            diagnostics.teacher_probabilities[agent_id]
        ),
        "domain_neural_decision_probability_mean": float(
            diagnostics.neural_probabilities[agent_id]
        ),
        "domain_bootstrapped_decision_probability_mean": float(
            diagnostics.bootstrapped_probabilities[agent_id]
        ),
        "domain_decision_bootstrap_teacher": diagnostics.teacher,
    }


def domain_decision_replay_micro_fields(
    diagnostics: DomainDecisionReplayDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent replay diagnostics using the aggregate field contract."""

    if diagnostics is None:
        return dict(_DOMAIN_DECISION_REPLAY_EMPTY_FIELDS)
    if agent_id >= diagnostics.candidate_mask.size:
        fields = dict(_DOMAIN_DECISION_REPLAY_EMPTY_FIELDS)
        fields["domain_decision_replay_weight"] = diagnostics.weight
        return fields

    candidate = bool(diagnostics.candidate_mask[agent_id])
    stable = bool(diagnostics.stable_teacher_mask[agent_id])
    objective = bool(diagnostics.objective_agreement_mask[agent_id])
    postsocial = bool(diagnostics.postsocial_improvement_mask[agent_id])
    return {
        "domain_decision_replay_weight": diagnostics.weight,
        "domain_decision_replay_candidate_rate": float(candidate),
        "domain_decision_replay_applied_rate": float(
            diagnostics.applied_mask[agent_id]
        ),
        "domain_decision_replay_rejected_unstable_teacher_rate": float(
            candidate and not stable
        ),
        "domain_decision_replay_rejected_objective_rate": float(
            candidate and stable and not objective
        ),
        "domain_decision_replay_rejected_postsocial_rate": float(
            candidate and stable and objective and not postsocial
        ),
        "domain_decision_replay_teacher_probability_mean": float(
            diagnostics.teacher_probabilities[agent_id]
        ),
        "domain_decision_replay_teacher": diagnostics.teacher,
    }


def domain_distill_bootstrap_micro_fields(
    diagnostics: DomainDistillBootstrapDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent distillation diagnostics using aggregate field names."""

    if diagnostics is None:
        return {
            "domain_distill_bootstrap_weight": "",
            "domain_distill_teacher": "",
            "domain_teacher_policy_bce_mean": "",
            "domain_teacher_policy_kl_mean": "",
            "domain_teacher_neural_probability_mean": "",
            "domain_teacher_probability_mean": "",
            "domain_teacher_neural_argmax_agreement": "",
            "domain_teacher_realized_action_agreement": "",
        }
    if agent_id >= diagnostics.teacher_probabilities.size:
        return {
            "domain_distill_bootstrap_weight": diagnostics.weight,
            "domain_distill_teacher": "",
            "domain_teacher_policy_bce_mean": "",
            "domain_teacher_policy_kl_mean": "",
            "domain_teacher_neural_probability_mean": "",
            "domain_teacher_probability_mean": "",
            "domain_teacher_neural_argmax_agreement": "",
            "domain_teacher_realized_action_agreement": "",
        }
    return {
        "domain_distill_bootstrap_weight": diagnostics.weight,
        "domain_distill_teacher": diagnostics.teacher,
        "domain_teacher_policy_bce_mean": float(diagnostics.bce[agent_id]),
        "domain_teacher_policy_kl_mean": float(diagnostics.kl[agent_id]),
        "domain_teacher_neural_probability_mean": float(
            diagnostics.neural_probabilities[agent_id]
        ),
        "domain_teacher_probability_mean": float(
            diagnostics.teacher_probabilities[agent_id]
        ),
        "domain_teacher_neural_argmax_agreement": float(
            diagnostics.argmax_agreement[agent_id]
        ),
        "domain_teacher_realized_action_agreement": float(
            diagnostics.realized_action_agreement[agent_id]
        ),
    }


def domain_teacher_alignment_micro_fields(
    diagnostics: DomainTeacherAlignmentDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent teacher-alignment diagnostics with aggregate field names."""

    if (
        diagnostics is None
        or diagnostics.teacher_pre_action.size == 0
        or agent_id >= diagnostics.teacher_pre_action.size
    ):
        return dict(_DOMAIN_TEACHER_ALIGNMENT_EMPTY_FIELDS)

    teacher_pre = diagnostics.teacher_pre_action
    bce_pre = teacher_policy_bce(diagnostics.pre_local, teacher_pre)
    bce_post_local = teacher_policy_bce(diagnostics.post_local, teacher_pre)
    bce_post_social = teacher_policy_bce(diagnostics.post_social, teacher_pre)
    kl_pre = teacher_policy_kl(diagnostics.pre_local, teacher_pre)
    kl_post_local = teacher_policy_kl(diagnostics.post_local, teacher_pre)
    kl_post_social = teacher_policy_kl(diagnostics.post_social, teacher_pre)
    teacher_action = int(teacher_pre[agent_id] >= 0.5)
    objective_agreement: float | str = ""
    objective_conflict: float | str = ""
    objective_margin: float | str = ""
    if diagnostics.objective_effective.size > agent_id:
        agreements, margins = objective_teacher_sign_alignment(
            diagnostics.objective_effective,
            teacher_pre,
        )
        objective_agreement = float(agreements[agent_id])
        objective_conflict = float(1.0 - agreements[agent_id])
        objective_margin = float(margins[agent_id])

    def value_at(values: np.ndarray) -> float | str:
        if agent_id >= values.size:
            return ""
        value = float(values[agent_id])
        return value if np.isfinite(value) else ""

    cosine = value_at(diagnostics.grad_cosines)
    candidate = (
        ""
        if agent_id >= diagnostics.distill_candidate_mask.size
        else bool(diagnostics.distill_candidate_mask[agent_id])
    )
    stable = (
        ""
        if agent_id >= diagnostics.distill_stable_teacher_mask.size
        else bool(diagnostics.distill_stable_teacher_mask[agent_id])
    )
    gradient_pass = (
        ""
        if agent_id >= diagnostics.distill_gradient_gate_mask.size
        else bool(diagnostics.distill_gradient_gate_mask[agent_id])
    )
    applied = (
        ""
        if agent_id >= diagnostics.distill_applied_mask.size
        else bool(diagnostics.distill_applied_mask[agent_id])
    )
    return {
        "domain_teacher_policy_bce_pre_local_mean": float(bce_pre[agent_id]),
        "domain_teacher_policy_bce_post_local_mean": float(
            bce_post_local[agent_id]
        ),
        "domain_teacher_policy_bce_post_social_mean": float(
            bce_post_social[agent_id]
        ),
        "domain_teacher_policy_kl_pre_local_mean": float(kl_pre[agent_id]),
        "domain_teacher_policy_kl_post_local_mean": float(kl_post_local[agent_id]),
        "domain_teacher_policy_kl_post_social_mean": float(
            kl_post_social[agent_id]
        ),
        "domain_teacher_bce_delta_local": float(
            bce_post_local[agent_id] - bce_pre[agent_id]
        ),
        "domain_teacher_bce_delta_social": float(
            bce_post_social[agent_id] - bce_post_local[agent_id]
        ),
        "domain_teacher_neural_argmax_agreement_pre_local": float(
            int(diagnostics.pre_local[agent_id] >= 0.5) == teacher_action
        ),
        "domain_teacher_neural_argmax_agreement_post_local": float(
            int(diagnostics.post_local[agent_id] >= 0.5) == teacher_action
        ),
        "domain_teacher_neural_argmax_agreement_post_social": float(
            int(diagnostics.post_social[agent_id] >= 0.5) == teacher_action
        ),
        "domain_teacher_realized_action_agreement": float(
            int(diagnostics.realized_actions[agent_id]) == teacher_action
        ),
        "domain_teacher_probability_pre_action_mean": float(teacher_pre[agent_id]),
        "domain_teacher_probability_post_action_mean": float(
            diagnostics.teacher_post_action[agent_id]
        ),
        "domain_teacher_target_shift_mean": float(
            abs(diagnostics.teacher_post_action[agent_id] - teacher_pre[agent_id])
        ),
        "domain_teacher_target_flip_rate": float(
            (diagnostics.teacher_post_action[agent_id] >= 0.5)
            != (teacher_pre[agent_id] >= 0.5)
        ),
        "domain_effective_advantage_teacher_sign_agreement": objective_agreement,
        "domain_effective_advantage_teacher_sign_conflict_rate": objective_conflict,
        "domain_effective_advantage_teacher_margin_mean": objective_margin,
        "domain_base_loss_mean": value_at(diagnostics.base_losses),
        "domain_distill_loss_mean": value_at(diagnostics.distill_losses),
        "domain_base_grad_norm_mean": value_at(diagnostics.base_grad_norms),
        "domain_distill_grad_norm_mean": value_at(diagnostics.distill_grad_norms),
        "domain_base_distill_grad_cosine_mean": cosine,
        "domain_base_distill_grad_cosine_negative_rate": (
            "" if cosine == "" else float(cosine < 0.0)
        ),
        "domain_distill_candidate_rate": "" if candidate == "" else float(candidate),
        "domain_distill_applied_rate": "" if applied == "" else float(applied),
        "domain_distill_rejected_unstable_teacher_rate": (
            ""
            if candidate == "" or stable == ""
            else float(bool(candidate) and not bool(stable))
        ),
        "domain_distill_rejected_gradient_rate": (
            ""
            if candidate == "" or stable == "" or gradient_pass == ""
            else float(bool(candidate) and bool(stable) and not bool(gradient_pass))
        ),
        "domain_teacher_alignment_teacher": diagnostics.teacher,
    }
