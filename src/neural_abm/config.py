"""Configuration models and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neural_abm.reputation import reputation_observation_extra_dim
from neural_abm.state_continuation import (
    BasinCreditConfig,
    DomainBootstrapConfig,
    StateContinuationObjectiveConfig,
)


class RunConfig(BaseModel):
    name: str
    seed: int
    output_dir: Path


class SimulationConfig(BaseModel):
    epochs: int = Field(gt=0)
    sync_mode: Literal["synchronous"] = "synchronous"
    device: str = "cpu"


class DataConfig(BaseModel):
    boundary: Literal["sine"] = "sine"
    label_noise: float = Field(ge=0.0, le=1.0)
    train_pool_size: int = Field(gt=0)
    probe_size: int = Field(gt=0)
    test_size: int = Field(gt=0)


class ModelConfig(BaseModel):
    input_dim: int = 2
    hidden_dim: int = Field(gt=0)
    output_dim: int = 2
    activation: Literal["relu"] = "relu"


class OptimizerConfig(BaseModel):
    name: Literal["adam"] = "adam"
    learning_rate: float = Field(gt=0.0)


class TrainingConfig(BaseModel):
    local_batch_size: int = Field(gt=0)
    local_steps_per_epoch: int = Field(gt=0)


class ShardGroupConfig(BaseModel):
    count: int = Field(gt=0)
    samples_per_agent: int = Field(gt=0)
    label_noise: float | None = Field(default=None, ge=0.0, le=1.0)


class ShardsConfig(BaseModel):
    policy: Literal["five_group_bias"] = "five_group_bias"
    groups: dict[str, ShardGroupConfig]


class AgentsConfig(BaseModel):
    count: int = Field(gt=0)
    init_mode: Literal["same_init", "independent_init"]
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    shards: ShardsConfig


class GraphConfig(BaseModel):
    type: Literal["watts_strogatz"]
    k: int = Field(gt=0)
    rewire_probability: float = Field(ge=0.0, le=1.0)


class CommunicationBudgetConfig(BaseModel):
    probe_predictions: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    scalar_summary: int = Field(gt=0)


class SocialConfig(BaseModel):
    mixer: Literal[
        "none",
        "output_average",
        "latent_average",
        "parameter_average",
        "parameter_aligned_average",
    ]
    peer_rule: Literal[
        "none",
        "state_similarity",
        "aligned_state_similarity",
        "latent_similarity",
        "output_similarity",
    ]
    alpha: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=-1.0, le=1.0)
    communication_budget: CommunicationBudgetConfig


class LoggingConfig(BaseModel):
    micro_state: bool = True
    interval: int = Field(gt=0)
    aggregate_metrics: bool = True
    probe_predictions: bool = False
    probe_prediction_interval: int = Field(default=1, gt=0)


class Toy1ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: AgentsConfig
    coordination: SocialConfig


class Toy1DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy1"] = "toy1"
    data: DataConfig
    graph: GraphConfig


class Toy1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy1ModelConfig
    domain: Toy1DomainConfig
    logging: LoggingConfig

    @property
    def data(self) -> DataConfig:
        return self.domain.data

    @property
    def agents(self) -> AgentsConfig:
        return self.model.agents

    @property
    def graph(self) -> GraphConfig:
        return self.domain.graph

    @property
    def coordination(self) -> SocialConfig:
        return self.model.coordination

    @property
    def social(self) -> SocialConfig:
        return self.model.coordination


class Toy2EnvironmentConfig(BaseModel):
    grid_width: int = Field(gt=1)
    grid_height: int = Field(gt=1)
    neighborhood: Literal["von_neumann"] = "von_neumann"
    periodic: bool = True
    initial_action_probability: float = Field(ge=0.0, le=1.0)
    reward_ema_decay: float = Field(ge=0.0, lt=1.0)
    entropy_beta: float = Field(ge=0.0)
    payoff_R: float = 3.0
    payoff_S: float = 0.0
    payoff_T: float = 5.0
    payoff_P: float = 1.0


class Toy2PayoffConfig(BaseModel):
    T: float = 5.0
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0


class Toy2GameConfig(BaseModel):
    family: Literal[
        "prisoner_dilemma",
        "snowdrift",
        "stag_hunt",
        "harmony",
    ] = "prisoner_dilemma"
    payoff: Toy2PayoffConfig = Field(default_factory=Toy2PayoffConfig)

    @model_validator(mode="after")
    def validate_family_inequalities(self) -> "Toy2GameConfig":
        payoff = self.payoff
        if self.family == "prisoner_dilemma":
            if not (payoff.T > payoff.R > payoff.P > payoff.S):
                raise ValueError("Prisoner's Dilemma requires T > R > P > S")
            if not (2.0 * payoff.R > payoff.T + payoff.S):
                raise ValueError("Prisoner's Dilemma requires 2R > T + S")
        elif self.family == "snowdrift":
            if not (payoff.T > payoff.R > payoff.S > payoff.P):
                raise ValueError("Snowdrift requires T > R > S > P")
        elif self.family == "stag_hunt":
            if not (payoff.R > payoff.T > payoff.P > payoff.S):
                raise ValueError("Stag Hunt requires R > T > P > S")
        elif self.family == "harmony":
            if not (payoff.R > payoff.T and payoff.S > payoff.P):
                raise ValueError("Harmony requires R > T and S > P")
        return self


def toy2_payoff_threshold(payoff: Toy2PayoffConfig) -> float:
    """Return the interior Stag-Hunt basin threshold implied by binary payoffs."""

    denominator = payoff.R - payoff.S - payoff.T + payoff.P
    if denominator == 0.0:
        raise ValueError("Payoff-threshold calibration denominator is zero")
    return (payoff.P - payoff.S) / denominator


def validate_toy2_payoff_threshold(payoff: Toy2PayoffConfig) -> float:
    threshold = toy2_payoff_threshold(payoff)
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Payoff-threshold calibration requires an interior threshold in "
            f"(0, 1); got {threshold:g}"
        )
    return threshold


class Toy2DecisionCalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "payoff_threshold"] = "none"
    strength: float = Field(default=4.0, gt=0.0)


class Toy2DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["sampled", "argmax"] = "sampled"
    action_temperature: float = Field(default=1.0, gt=0.0)
    exploration_epsilon: float = Field(default=0.0, ge=0.0, le=1.0)
    terminal_argmax_epochs: int = Field(default=0, ge=0)
    calibration: Toy2DecisionCalibrationConfig = Field(
        default_factory=Toy2DecisionCalibrationConfig
    )


class Toy2PolicyDomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_update_rule: Literal[
        "sampled_policy_gradient",
        "counterfactual_advantage",
    ] = "sampled_policy_gradient"
    neural_peer_mode: Literal["spatial", "well_mixed"] = "spatial"
    interaction_mode: Literal["spatial", "well_mixed_resampled"] = "spatial"
    payoff_transform: Literal["linear", "tanh"] = "linear"
    objective: StateContinuationObjectiveConfig = Field(
        default_factory=StateContinuationObjectiveConfig
    )
    bootstrap: DomainBootstrapConfig = Field(default_factory=DomainBootstrapConfig)
    basin_credit: BasinCreditConfig = Field(default_factory=BasinCreditConfig)


class Toy2PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal[
        "neural_policy",
        "fermi_imitation",
        "rd_well_mixed",
        "reputation_imitation",
    ] = "neural_policy"
    learning_enabled: bool = True
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    selection_strength: float = Field(default=1.0, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0)
    neural_update_backend: Literal["loop", "batched", "tensor_batched", "auto"] = "loop"
    decision: Toy2DecisionConfig = Field(default_factory=Toy2DecisionConfig)
    domain: Toy2PolicyDomainConfig = Field(default_factory=Toy2PolicyDomainConfig)

    @model_validator(mode="after")
    def canonicalize_inactive_decision_fields(self) -> "Toy2PolicyConfig":
        if self.rule != "neural_policy":
            self.decision = Toy2DecisionConfig()
            self.neural_update_backend = "loop"
        elif self.decision.mode == "argmax":
            self.decision.action_temperature = 1.0
            self.decision.exploration_epsilon = 0.0
            self.decision.terminal_argmax_epochs = 0
            self.decision.calibration = Toy2DecisionCalibrationConfig()
        if (
            self.rule == "neural_policy"
            and self.domain.objective.uses_state_continuation()
        ):
            if self.neural_update_backend != "loop":
                raise ValueError(
                    "Toy 2 state_continuation objective requires "
                    "policy.neural_update_backend='loop'"
                )
            if self.domain.local_update_rule != "counterfactual_advantage":
                raise ValueError(
                    "Toy 2 state_continuation objective requires "
                    "policy.domain.local_update_rule='counterfactual_advantage'"
                )
        if (
            self.domain.bootstrap.enabled
            or self.domain.bootstrap.decision_enabled
            or self.domain.bootstrap.distill_enabled
            or self.domain.bootstrap.replay_enabled
        ):
            if self.rule != "neural_policy":
                raise ValueError(
                    "Toy 2 domain bootstrap requires policy.rule='neural_policy'"
                )
            if self.neural_update_backend != "loop":
                raise ValueError(
                    "Toy 2 domain bootstrap requires "
                    "policy.neural_update_backend='loop'"
                )
            if not self.domain.objective.uses_state_continuation():
                raise ValueError(
                    "Toy 2 domain bootstrap requires state_continuation objective"
                )
            if self.domain.local_update_rule != "counterfactual_advantage":
                raise ValueError(
                    "Toy 2 domain bootstrap requires "
                    "policy.domain.local_update_rule='counterfactual_advantage'"
                )
        if self.domain.basin_credit.enabled:
            if self.rule != "neural_policy":
                raise ValueError(
                    "Toy 2 basin credit requires policy.rule='neural_policy'"
                )
            if self.neural_update_backend != "loop":
                raise ValueError(
                    "Toy 2 basin credit requires policy.neural_update_backend='loop'"
                )
            if not self.domain.objective.uses_state_continuation():
                raise ValueError(
                    "Toy 2 basin credit requires state_continuation objective"
                )
            if self.domain.local_update_rule != "counterfactual_advantage":
                raise ValueError(
                    "Toy 2 basin credit requires "
                    "policy.domain.local_update_rule='counterfactual_advantage'"
                )
        return self


class Toy2AgentsConfig(BaseModel):
    init_mode: Literal["same_init", "independent_init"]
    model: ModelConfig
    optimizer: OptimizerConfig
    policy_prior_action_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class Toy2CoordinationConfig(BaseModel):
    mixer: Literal["none", "output_average"]
    peer_rule: Literal["none", "output_similarity"]
    alpha: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=-1.0, le=1.0)
    confidence_weighting: Literal["none", "peer", "peer_direction"] = "none"
    confidence_weight_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_weight_power: float = Field(default=1.0, gt=0.0)
    confidence_tail_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_tail_min_policy_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_tail_min_action_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    commitment_enabled: bool = False
    commitment_min_policy_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    commitment_min_action_streak: int = Field(default=1, ge=1)
    commitment_requires_direction: bool = True
    commitment_min_direction: float = 0.0
    commitment_exit_policy_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    commitment_exit_on_negative_direction: bool = True
    precommitment_enabled: bool = False
    precommitment_min_policy_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_min_evidence: float = Field(default=1.0, ge=0.0)
    precommitment_evidence_increment: float = Field(default=1.0, ge=0.0)
    precommitment_evidence_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    precommitment_requires_direction: bool = True
    precommitment_min_direction: float = 0.0
    precommitment_direction_source: Literal[
        "social",
        "local_threshold",
        "readiness_augmented_threshold",
    ] = "social"
    precommitment_readiness_direction_weight: float = Field(default=1.0, ge=0.0)
    precommitment_decision_feedback_enabled: bool = False
    precommitment_decision_feedback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_social_feedback_enabled: bool = False
    precommitment_social_feedback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_peer_evidence_enabled: bool = False
    precommitment_peer_evidence_weight: float = Field(default=0.0, ge=0.0)
    precommitment_peer_readiness_aggregation: Literal["mean", "max"] = "mean"
    revision_operator_enabled: bool = False
    revision_operator_source: Literal["policy_probability"] = "policy_probability"
    communication_budget: CommunicationBudgetConfig


class ReputationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    decay: float = Field(default=0.9, ge=0.0, lt=1.0)
    peer_rule: Literal["spatial", "well_mixed"] = "spatial"
    temperature: float = Field(default=1.0, gt=0.0)
    noise: float = Field(default=0.0, ge=0.0)
    observation_mode: Literal["none", "self_neighbor_mean"] = "none"

    @model_validator(mode="after")
    def validate_observation_mode(self) -> "ReputationConfig":
        if self.observation_mode != "none" and not self.enabled:
            raise ValueError("reputation observation_mode requires reputation.enabled")
        return self


class MobilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    rate: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_pool_size: int = Field(default=8, gt=0)
    selection_rule: Literal["local_quality"] = "local_quality"
    move_cost: float = Field(default=0.0, ge=0.0)


class BinaryStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reputation: ReputationConfig = Field(default_factory=ReputationConfig)
    mobility: MobilityConfig = Field(default_factory=MobilityConfig)


class Toy2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: "Toy2ModelConfig"
    domain: "Toy2DomainConfig"
    logging: LoggingConfig

    @model_validator(mode="after")
    def sync_legacy_environment_payoff_fields(self) -> "Toy2Config":
        payoff = self.game.payoff
        self.environment.payoff_T = payoff.T
        self.environment.payoff_R = payoff.R
        self.environment.payoff_P = payoff.P
        self.environment.payoff_S = payoff.S
        return self

    @model_validator(mode="after")
    def validate_decision_calibration(self) -> "Toy2Config":
        decision = self.policy.decision
        if (
            self.policy.rule == "neural_policy"
            and decision.mode == "sampled"
            and decision.calibration.mode == "payoff_threshold"
        ):
            validate_toy2_payoff_threshold(self.game.payoff)
        return self

    @model_validator(mode="after")
    def validate_reputation_path(self) -> "Toy2Config":
        if self.policy.rule == "neural_policy":
            expected_input_dim = 6 + reputation_observation_extra_dim(
                self.state.reputation.observation_mode
            )
            if self.agents.model.input_dim != expected_input_dim:
                raise ValueError(
                    f"Toy 2 neural_policy expects model.input_dim={expected_input_dim}"
                )
        if (
            self.policy.rule == "reputation_imitation"
            and not self.state.reputation.enabled
        ):
            raise ValueError("Toy 2 reputation_imitation requires reputation.enabled")
        return self

    @property
    def game(self) -> Toy2GameConfig:
        return self.domain.game

    @property
    def policy(self) -> Toy2PolicyConfig:
        return self.model.policy

    @property
    def environment(self) -> Toy2EnvironmentConfig:
        return self.domain.environment

    @property
    def agents(self) -> Toy2AgentsConfig:
        return self.model.agents

    @property
    def coordination(self) -> Toy2CoordinationConfig:
        return self.model.coordination

    @property
    def state(self) -> BinaryStateConfig:
        return self.model.state


class Toy2ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy2PolicyConfig = Field(default_factory=Toy2PolicyConfig)
    agents: Toy2AgentsConfig
    coordination: Toy2CoordinationConfig
    state: BinaryStateConfig = Field(default_factory=BinaryStateConfig)


class Toy2DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy2"] = "toy2"
    environment: Toy2EnvironmentConfig
    game: Toy2GameConfig = Field(default_factory=Toy2GameConfig)


class Toy3EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opinion_min: float = -1.0
    opinion_max: float = 1.0
    initial_opinion_mode: Literal["two_clusters", "uniform"] = "two_clusters"
    cluster_centers: tuple[float, float] = (-0.4, 0.4)
    cluster_std: float = Field(default=0.08, ge=0.0)

    @model_validator(mode="after")
    def validate_opinion_bounds(self) -> "Toy3EnvironmentConfig":
        if self.opinion_min >= self.opinion_max:
            raise ValueError("Toy 3 opinion_min must be less than opinion_max")
        for center in self.cluster_centers:
            if not self.opinion_min <= center <= self.opinion_max:
                raise ValueError("Toy 3 cluster centers must lie within opinion bounds")
        return self


class Toy3DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    update_rule: Literal["hk", "deffuant", "neural_policy"] = "hk"
    confidence_threshold: float = Field(default=0.35, gt=0.0)
    influence_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    deffuant_mu: float = Field(default=0.5, ge=0.0, le=0.5)
    neural_delta_scale: float = Field(default=0.25, gt=0.0)
    neural_learning_rate: float = Field(default=0.01, ge=0.0)


class Toy3AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, gt=1)
    init_mode: Literal["same_init", "independent_init"] = "independent_init"
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(input_dim=6, hidden_dim=16, output_dim=1)
    )
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(learning_rate=0.01)
    )


class Toy3GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["watts_strogatz"] = "watts_strogatz"
    k: int = Field(default=6, gt=0)
    rewire_probability: float = Field(default=0.1, ge=0.0, le=1.0)


class Toy3SocialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "bounded_confidence", "output_similarity"] = (
        "bounded_confidence"
    )
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class Toy3RewiringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold: float = Field(default=0.8, gt=0.0)
    rate: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_pool_size: int = Field(default=10, gt=0)


class Toy3ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy3DynamicsConfig = Field(default_factory=Toy3DynamicsConfig)
    agents: Toy3AgentsConfig = Field(default_factory=Toy3AgentsConfig)
    coordination: Toy3SocialConfig = Field(default_factory=Toy3SocialConfig)


class Toy3DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy3"] = "toy3"
    environment: Toy3EnvironmentConfig = Field(default_factory=Toy3EnvironmentConfig)
    graph: Toy3GraphConfig = Field(default_factory=Toy3GraphConfig)
    rewiring: Toy3RewiringConfig = Field(default_factory=Toy3RewiringConfig)


class Toy3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy3ModelConfig = Field(default_factory=Toy3ModelConfig)
    domain: Toy3DomainConfig = Field(default_factory=Toy3DomainConfig)
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy3_config(self) -> "Toy3Config":
        opinion_range = self.environment.opinion_max - self.environment.opinion_min
        if self.dynamics.confidence_threshold > opinion_range:
            raise ValueError("Toy 3 confidence_threshold exceeds opinion range")
        if self.rewiring.threshold > opinion_range:
            raise ValueError("Toy 3 rewiring threshold exceeds opinion range")
        if self.graph.k >= self.agents.count:
            raise ValueError("Toy 3 graph.k must be less than agents.count")
        if self.dynamics.update_rule == "neural_policy":
            if self.agents.model.input_dim != 6:
                raise ValueError("Toy 3 neural_policy expects model.input_dim=6")
            if self.agents.model.output_dim != 1:
                raise ValueError("Toy 3 neural_policy expects model.output_dim=1")
        return self

    @property
    def environment(self) -> Toy3EnvironmentConfig:
        return self.domain.environment

    @property
    def dynamics(self) -> Toy3DynamicsConfig:
        return self.model.policy

    @property
    def policy(self) -> Toy3DynamicsConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy3AgentsConfig:
        return self.model.agents

    @property
    def graph(self) -> Toy3GraphConfig:
        return self.domain.graph

    @property
    def social(self) -> Toy3SocialConfig:
        return self.model.coordination

    @property
    def coordination(self) -> Toy3SocialConfig:
        return self.model.coordination

    @property
    def rewiring(self) -> Toy3RewiringConfig:
        return self.domain.rewiring


class Toy4EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_width: int = Field(default=10, gt=1)
    grid_height: int = Field(default=10, gt=1)
    initial_action_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    reward_ema_decay: float = Field(default=0.90, ge=0.0, lt=1.0)
    entropy_beta: float = Field(default=0.01, ge=0.0)
    resource_enabled: bool = False
    resource_initial: float = Field(default=100.0, ge=0.0)
    resource_carrying_capacity: float = Field(default=100.0, gt=0.0)
    resource_recovery_rate: float = Field(default=0.05, ge=0.0)
    resource_extraction_per_defector: float = Field(default=1.0, ge=0.0)
    resource_extraction_heterogeneity: float = Field(default=0.0, ge=0.0, le=1.0)
    resource_extraction_heterogeneity_mode: Literal["none", "checkerboard"] = "none"
    resource_observation_mode: Literal["global", "hidden", "local_sustain"] = "global"
    resource_collapse_threshold: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_resource_bounds(self) -> "Toy4EnvironmentConfig":
        if self.resource_initial > self.resource_carrying_capacity:
            raise ValueError(
                "Toy 4 resource_initial must be <= resource_carrying_capacity"
            )
        if self.resource_collapse_threshold > self.resource_carrying_capacity:
            raise ValueError(
                "Toy 4 resource_collapse_threshold must be <= carrying capacity"
            )
        if (
            self.resource_extraction_heterogeneity > 0.0
            and self.resource_extraction_heterogeneity_mode == "none"
        ):
            raise ValueError(
                "Toy 4 resource_extraction_heterogeneity requires a non-none mode"
            )
        return self


class Toy4GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    multiplier: float = Field(default=1.6, gt=0.0)
    contribution_cost: float = Field(default=1.0, ge=0.0)
    group_mode: Literal["local_neighborhood"] = "local_neighborhood"


class Toy4DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["sampled", "argmax"] = "sampled"
    action_temperature: float = Field(default=1.0, gt=0.0)
    exploration_epsilon: float = Field(default=0.0, ge=0.0, le=1.0)
    terminal_argmax_epochs: int = Field(default=0, ge=0)


class Toy4PolicyDomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: StateContinuationObjectiveConfig = Field(
        default_factory=StateContinuationObjectiveConfig
    )
    resource_environment_pressure_weight: float = Field(default=1.0, ge=0.0)
    resource_environment_lookahead_weight: float = Field(default=0.0, ge=0.0)
    resource_environment_threshold_weight: float = Field(default=0.0, ge=0.0)
    resource_environment_threshold_scope: Literal["population", "local"] = (
        "population"
    )
    bootstrap: DomainBootstrapConfig = Field(default_factory=DomainBootstrapConfig)
    basin_credit: BasinCreditConfig = Field(default_factory=BasinCreditConfig)


class Toy4PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal[
        "neural_policy",
        "imitation",
        "reputation_imitation",
    ] = "neural_policy"
    learning_enabled: bool = True
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    selection_strength: float = Field(default=1.0, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0)
    neural_update_backend: Literal["loop", "batched", "tensor_batched", "auto"] = "loop"
    decision: Toy4DecisionConfig = Field(default_factory=Toy4DecisionConfig)
    domain: Toy4PolicyDomainConfig = Field(default_factory=Toy4PolicyDomainConfig)

    @model_validator(mode="after")
    def canonicalize_inactive_decision_fields(self) -> "Toy4PolicyConfig":
        if self.rule != "neural_policy":
            self.decision = Toy4DecisionConfig()
            self.neural_update_backend = "loop"
        elif self.decision.mode == "argmax":
            self.decision.action_temperature = 1.0
            self.decision.exploration_epsilon = 0.0
            self.decision.terminal_argmax_epochs = 0
        if (
            self.rule == "neural_policy"
            and self.domain.objective.uses_state_continuation()
            and self.neural_update_backend != "loop"
        ):
            raise ValueError(
                "Toy 4 state_continuation objective requires "
                "policy.neural_update_backend='loop'"
            )
        if (
            self.domain.bootstrap.enabled
            or self.domain.bootstrap.decision_enabled
            or self.domain.bootstrap.distill_enabled
            or self.domain.bootstrap.replay_enabled
        ):
            if self.rule != "neural_policy":
                raise ValueError(
                    "Toy 4 domain bootstrap requires policy.rule='neural_policy'"
                )
            if self.neural_update_backend != "loop":
                raise ValueError(
                    "Toy 4 domain bootstrap requires "
                    "policy.neural_update_backend='loop'"
                )
            if not self.domain.objective.uses_state_continuation():
                raise ValueError(
                    "Toy 4 domain bootstrap requires state_continuation objective"
                )
        if self.domain.basin_credit.enabled:
            if self.rule != "neural_policy":
                raise ValueError(
                    "Toy 4 basin credit requires policy.rule='neural_policy'"
                )
            if self.neural_update_backend != "loop":
                raise ValueError(
                    "Toy 4 basin credit requires policy.neural_update_backend='loop'"
                )
            if not self.domain.objective.uses_state_continuation():
                raise ValueError(
                    "Toy 4 basin credit requires state_continuation objective"
                )
        return self


class Toy4AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    init_mode: Literal["same_init", "independent_init"] = "independent_init"
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(input_dim=6, hidden_dim=16, output_dim=2)
    )
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(learning_rate=0.01)
    )


class Toy4GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["grid"] = "grid"
    neighborhood: Literal["von_neumann"] = "von_neumann"
    periodic: bool = True


class Toy4CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence_weighting: Literal["none", "peer", "peer_direction"] = "none"
    confidence_weight_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_weight_power: float = Field(default=1.0, gt=0.0)
    confidence_tail_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_tail_min_policy_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_tail_min_action_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    commitment_enabled: bool = False
    commitment_min_policy_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    commitment_min_action_streak: int = Field(default=1, ge=1)
    commitment_requires_direction: bool = True
    commitment_min_direction: float = 0.0
    commitment_exit_policy_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    commitment_exit_on_negative_direction: bool = True
    precommitment_enabled: bool = False
    precommitment_min_policy_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_min_evidence: float = Field(default=1.0, ge=0.0)
    precommitment_evidence_increment: float = Field(default=1.0, ge=0.0)
    precommitment_evidence_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    precommitment_requires_direction: bool = True
    precommitment_min_direction: float = 0.0
    precommitment_direction_source: Literal[
        "social",
        "local_threshold",
        "readiness_augmented_threshold",
    ] = "social"
    precommitment_readiness_direction_weight: float = Field(default=1.0, ge=0.0)
    precommitment_decision_feedback_enabled: bool = False
    precommitment_decision_feedback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_social_feedback_enabled: bool = False
    precommitment_social_feedback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_peer_evidence_enabled: bool = False
    precommitment_peer_evidence_weight: float = Field(default=0.0, ge=0.0)
    precommitment_peer_readiness_aggregation: Literal["mean", "max"] = "mean"
    revision_operator_enabled: bool = False
    revision_operator_source: Literal["policy_probability"] = "policy_probability"


class Toy4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: "Toy4ModelConfig" = Field(default_factory=lambda: Toy4ModelConfig())
    domain: "Toy4DomainConfig" = Field(default_factory=lambda: Toy4DomainConfig())
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy4_config(self) -> "Toy4Config":
        if self.policy.rule == "neural_policy":
            expected_input_dim = 6 + reputation_observation_extra_dim(
                self.state.reputation.observation_mode
            )
            if self.agents.model.input_dim != expected_input_dim:
                raise ValueError(
                    f"Toy 4 neural_policy expects model.input_dim={expected_input_dim}"
                )
            if self.agents.model.output_dim != 2:
                raise ValueError("Toy 4 neural_policy expects model.output_dim=2")
        if (
            self.policy.rule == "reputation_imitation"
            and not self.state.reputation.enabled
        ):
            raise ValueError("Toy 4 reputation_imitation requires reputation.enabled")
        return self

    @property
    def agent_count(self) -> int:
        return self.environment.grid_width * self.environment.grid_height

    @property
    def environment(self) -> Toy4EnvironmentConfig:
        return self.domain.environment

    @property
    def game(self) -> Toy4GameConfig:
        return self.domain.game

    @property
    def policy(self) -> Toy4PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy4AgentsConfig:
        return self.model.agents

    @property
    def graph(self) -> Toy4GraphConfig:
        return self.domain.graph

    @property
    def coordination(self) -> Toy4CoordinationConfig:
        return self.model.coordination

    @property
    def state(self) -> BinaryStateConfig:
        return self.model.state


class Toy4ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy4PolicyConfig = Field(default_factory=Toy4PolicyConfig)
    agents: Toy4AgentsConfig = Field(default_factory=Toy4AgentsConfig)
    coordination: Toy4CoordinationConfig = Field(default_factory=Toy4CoordinationConfig)
    state: BinaryStateConfig = Field(default_factory=BinaryStateConfig)


class Toy4DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy4"] = "toy4"
    environment: Toy4EnvironmentConfig = Field(default_factory=Toy4EnvironmentConfig)
    game: Toy4GameConfig = Field(default_factory=Toy4GameConfig)
    graph: Toy4GraphConfig = Field(default_factory=Toy4GraphConfig)


class Toy5EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_action_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    seed_selection: Literal["random", "first_agent"] = "random"
    threshold_mode: Literal["homogeneous", "heterogeneous"] = "homogeneous"
    homogeneous_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    heterogeneous_threshold_low: float = Field(default=0.15, ge=0.0, le=1.0)
    heterogeneous_threshold_high: float = Field(default=0.55, ge=0.0, le=1.0)
    simple_contagion_probability: float = Field(default=0.08, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "Toy5EnvironmentConfig":
        if self.heterogeneous_threshold_low > self.heterogeneous_threshold_high:
            raise ValueError(
                "Toy 5 heterogeneous_threshold_low must be <= "
                "heterogeneous_threshold_high"
            )
        return self


class Toy5DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["sampled", "argmax"] = "sampled"
    action_temperature: float = Field(default=1.0, gt=0.0)
    exploration_epsilon: float = Field(default=0.0, ge=0.0, le=1.0)


class Toy5PolicyDomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_update_rule: Literal["adoption_utility", "threshold_target"] = (
        "adoption_utility"
    )
    repeated_exposure_decay: float = Field(default=0.0, ge=0.0, lt=1.0)
    adoption_is_absorbing: bool = True


class Toy5PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal[
        "simple_contagion",
        "complex_threshold",
        "neural_policy",
        "reputation_imitation",
    ] = "complex_threshold"
    learning_enabled: bool = True
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    neural_update_backend: Literal["loop", "batched", "tensor_batched", "auto"] = "loop"
    decision: Toy5DecisionConfig = Field(default_factory=Toy5DecisionConfig)
    domain: Toy5PolicyDomainConfig = Field(default_factory=Toy5PolicyDomainConfig)

    @model_validator(mode="after")
    def canonicalize_inactive_decision_fields(self) -> "Toy5PolicyConfig":
        if self.rule != "neural_policy":
            self.decision = Toy5DecisionConfig()
            self.neural_update_backend = "loop"
        elif self.decision.mode == "argmax":
            self.decision.action_temperature = 1.0
            self.decision.exploration_epsilon = 0.0
        return self


class Toy5AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, gt=1)
    init_mode: Literal["same_init", "independent_init"] = "independent_init"
    policy_prior_action_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(input_dim=6, hidden_dim=16, output_dim=2)
    )
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(learning_rate=0.01)
    )


class Toy5GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["watts_strogatz"] = "watts_strogatz"
    k: int = Field(default=6, gt=0)
    rewire_probability: float = Field(default=0.1, ge=0.0, le=1.0)


class Toy5CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence_weighting: Literal["none", "peer", "peer_direction"] = "none"
    confidence_weight_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_weight_power: float = Field(default=1.0, gt=0.0)
    confidence_tail_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_tail_min_policy_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_tail_min_action_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    commitment_enabled: bool = False
    commitment_min_policy_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    commitment_min_action_streak: int = Field(default=1, ge=1)
    commitment_requires_direction: bool = True
    commitment_min_direction: float = 0.0
    commitment_exit_policy_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    commitment_exit_on_negative_direction: bool = True
    precommitment_enabled: bool = False
    precommitment_min_policy_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_min_evidence: float = Field(default=1.0, ge=0.0)
    precommitment_evidence_increment: float = Field(default=1.0, ge=0.0)
    precommitment_evidence_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    precommitment_requires_direction: bool = True
    precommitment_min_direction: float = 0.0
    precommitment_direction_source: Literal[
        "social",
        "local_threshold",
        "readiness_augmented_threshold",
        "readiness_augmented_threshold_with_action_anchor",
        "readiness_exposure_with_action_anchor",
    ] = "social"
    precommitment_readiness_direction_weight: float = Field(default=1.0, ge=0.0)
    precommitment_decision_feedback_enabled: bool = False
    precommitment_decision_feedback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_social_feedback_enabled: bool = False
    precommitment_social_feedback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    precommitment_peer_evidence_enabled: bool = False
    precommitment_peer_evidence_weight: float = Field(default=0.0, ge=0.0)
    precommitment_peer_readiness_aggregation: Literal["mean", "max"] = "mean"


class Toy5Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: "Toy5ModelConfig" = Field(default_factory=lambda: Toy5ModelConfig())
    domain: "Toy5DomainConfig" = Field(default_factory=lambda: Toy5DomainConfig())
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy5_config(self) -> "Toy5Config":
        if self.graph.k >= self.agents.count:
            raise ValueError("Toy 5 graph.k must be less than agents.count")
        if self.policy.rule == "neural_policy":
            expected_input_dim = 6 + reputation_observation_extra_dim(
                self.state.reputation.observation_mode
            )
            if self.agents.model.input_dim != expected_input_dim:
                raise ValueError(
                    f"Toy 5 neural_policy expects model.input_dim={expected_input_dim}"
                )
            if self.agents.model.output_dim != 2:
                raise ValueError("Toy 5 neural_policy expects model.output_dim=2")
        if (
            self.policy.rule == "reputation_imitation"
            and not self.state.reputation.enabled
        ):
            raise ValueError("Toy 5 reputation_imitation requires reputation.enabled")
        return self

    @property
    def environment(self) -> Toy5EnvironmentConfig:
        return self.domain.environment

    @property
    def policy(self) -> Toy5PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy5AgentsConfig:
        return self.model.agents

    @property
    def graph(self) -> Toy5GraphConfig:
        return self.domain.graph

    @property
    def coordination(self) -> Toy5CoordinationConfig:
        return self.model.coordination

    @property
    def state(self) -> BinaryStateConfig:
        return self.model.state


class Toy5ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy5PolicyConfig = Field(default_factory=Toy5PolicyConfig)
    agents: Toy5AgentsConfig = Field(default_factory=Toy5AgentsConfig)
    coordination: Toy5CoordinationConfig = Field(default_factory=Toy5CoordinationConfig)
    state: BinaryStateConfig = Field(default_factory=BinaryStateConfig)


class Toy5DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy5"] = "toy5"
    environment: Toy5EnvironmentConfig = Field(default_factory=Toy5EnvironmentConfig)
    graph: Toy5GraphConfig = Field(default_factory=Toy5GraphConfig)


class Toy6EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_width: int = Field(default=10, gt=1)
    grid_height: int = Field(default=10, gt=1)
    initial_strategy_probabilities: list[float] | None = None
    reward_ema_decay: float = Field(default=0.90, ge=0.0, lt=1.0)


class Toy6PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal["categorical_learning"] = "categorical_learning"
    learning_rate: float = Field(default=0.15, ge=0.0)
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    decision_mode: Literal["sampled", "argmax"] = "sampled"


class Toy6AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    init_mode: Literal["same_init", "independent_init"] = "independent_init"
    logit_noise: float = Field(default=0.1, ge=0.0)


class Toy6CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class Toy6GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["grid"] = "grid"
    neighborhood: Literal["von_neumann"] = "von_neumann"
    periodic: bool = True


class Toy6GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_count: int = Field(default=3, ge=3)
    win_payoff: float = 1.0
    loss_payoff: float = -1.0
    draw_payoff: float = 0.0


class Toy6ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy6PolicyConfig = Field(default_factory=Toy6PolicyConfig)
    agents: Toy6AgentsConfig = Field(default_factory=Toy6AgentsConfig)
    coordination: Toy6CoordinationConfig = Field(default_factory=Toy6CoordinationConfig)


class Toy6DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy6"] = "toy6"
    environment: Toy6EnvironmentConfig = Field(default_factory=Toy6EnvironmentConfig)
    game: Toy6GameConfig = Field(default_factory=Toy6GameConfig)
    graph: Toy6GraphConfig = Field(default_factory=Toy6GraphConfig)


class Toy6Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy6ModelConfig = Field(default_factory=Toy6ModelConfig)
    domain: Toy6DomainConfig = Field(default_factory=Toy6DomainConfig)
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy6_config(self) -> "Toy6Config":
        probabilities = self.environment.initial_strategy_probabilities
        if probabilities is not None:
            if len(probabilities) != self.game.strategy_count:
                raise ValueError(
                    "Toy 6 initial_strategy_probabilities length must match "
                    "game.strategy_count"
                )
            if any(value < 0.0 for value in probabilities):
                raise ValueError(
                    "Toy 6 initial_strategy_probabilities must be non-negative"
                )
            total = sum(probabilities)
            if total <= 0.0:
                raise ValueError(
                    "Toy 6 initial_strategy_probabilities must have positive mass"
                )
        return self

    @property
    def agent_count(self) -> int:
        return self.environment.grid_width * self.environment.grid_height

    @property
    def environment(self) -> Toy6EnvironmentConfig:
        return self.domain.environment

    @property
    def game(self) -> Toy6GameConfig:
        return self.domain.game

    @property
    def graph(self) -> Toy6GraphConfig:
        return self.domain.graph

    @property
    def policy(self) -> Toy6PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy6AgentsConfig:
        return self.model.agents

    @property
    def coordination(self) -> Toy6CoordinationConfig:
        return self.model.coordination


class Toy7EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_initial: float = Field(default=80.0, ge=0.0)
    resource_carrying_capacity: float = Field(default=100.0, gt=0.0)
    resource_recovery_rate: float = Field(default=0.05, ge=0.0)
    extraction_scale: float = Field(default=8.0, ge=0.0)
    extraction_cost: float = Field(default=0.35, ge=0.0)
    initial_intensity_mean: float = Field(default=0.35, ge=0.0, le=1.0)
    initial_intensity_std: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def validate_resource_initial(self) -> "Toy7EnvironmentConfig":
        if self.resource_initial > self.resource_carrying_capacity:
            raise ValueError(
                "Toy 7 resource_initial must be <= resource_carrying_capacity"
            )
        return self


class Toy7PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal["adaptive_intensity"] = "adaptive_intensity"
    learning_rate: float = Field(default=0.20, ge=0.0)
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    exploration_std: float = Field(default=0.02, ge=0.0)
    reward_ema_decay: float = Field(default=0.90, ge=0.0, lt=1.0)


class Toy7AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, gt=1)
    init_mode: Literal["same_init", "independent_init"] = "independent_init"


class Toy7CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class Toy7GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["watts_strogatz"] = "watts_strogatz"
    k: int = Field(default=6, gt=0)
    rewire_probability: float = Field(default=0.1, ge=0.0, le=1.0)


class Toy7ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy7PolicyConfig = Field(default_factory=Toy7PolicyConfig)
    agents: Toy7AgentsConfig = Field(default_factory=Toy7AgentsConfig)
    coordination: Toy7CoordinationConfig = Field(default_factory=Toy7CoordinationConfig)


class Toy7DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy7"] = "toy7"
    environment: Toy7EnvironmentConfig = Field(default_factory=Toy7EnvironmentConfig)
    graph: Toy7GraphConfig = Field(default_factory=Toy7GraphConfig)


class Toy7Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy7ModelConfig = Field(default_factory=Toy7ModelConfig)
    domain: Toy7DomainConfig = Field(default_factory=Toy7DomainConfig)
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy7_config(self) -> "Toy7Config":
        if self.graph.k >= self.agents.count:
            raise ValueError("Toy 7 graph.k must be less than agents.count")
        return self

    @property
    def environment(self) -> Toy7EnvironmentConfig:
        return self.domain.environment

    @property
    def graph(self) -> Toy7GraphConfig:
        return self.domain.graph

    @property
    def policy(self) -> Toy7PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy7AgentsConfig:
        return self.model.agents

    @property
    def coordination(self) -> Toy7CoordinationConfig:
        return self.model.coordination


class Toy8EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_active_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    initial_failed_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    base_activation_rate: float = Field(default=0.02, ge=0.0)
    peer_activation_rate: float = Field(default=0.30, ge=0.0)
    failure_rate: float = Field(default=0.03, ge=0.0)
    overload_failure_rate: float = Field(default=0.08, ge=0.0)
    recovery_rate: float = Field(default=0.01, ge=0.0)
    max_time: float = Field(default=50.0, gt=0.0)

    @model_validator(mode="after")
    def validate_initial_fractions(self) -> "Toy8EnvironmentConfig":
        if self.initial_active_fraction + self.initial_failed_fraction > 1.0:
            raise ValueError(
                "Toy 8 initial_active_fraction + initial_failed_fraction must be <= 1"
            )
        return self


class Toy8PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal["event_hazard"] = "event_hazard"


class Toy8AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, gt=1)


class Toy8CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class Toy8GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["watts_strogatz"] = "watts_strogatz"
    k: int = Field(default=4, gt=0)
    rewire_probability: float = Field(default=0.1, ge=0.0, le=1.0)


class Toy8ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy8PolicyConfig = Field(default_factory=Toy8PolicyConfig)
    agents: Toy8AgentsConfig = Field(default_factory=Toy8AgentsConfig)
    coordination: Toy8CoordinationConfig = Field(default_factory=Toy8CoordinationConfig)


class Toy8DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy8"] = "toy8"
    environment: Toy8EnvironmentConfig = Field(default_factory=Toy8EnvironmentConfig)
    graph: Toy8GraphConfig = Field(default_factory=Toy8GraphConfig)


class Toy8Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy8ModelConfig = Field(default_factory=Toy8ModelConfig)
    domain: Toy8DomainConfig = Field(default_factory=Toy8DomainConfig)
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy8_config(self) -> "Toy8Config":
        if self.graph.k >= self.agents.count:
            raise ValueError("Toy 8 graph.k must be less than agents.count")
        env = self.environment
        if (
            env.base_activation_rate
            + env.peer_activation_rate
            + env.failure_rate
            + env.overload_failure_rate
            + env.recovery_rate
            <= 0.0
        ):
            raise ValueError("Toy 8 requires at least one positive event rate")
        return self

    @property
    def environment(self) -> Toy8EnvironmentConfig:
        return self.domain.environment

    @property
    def graph(self) -> Toy8GraphConfig:
        return self.domain.graph

    @property
    def policy(self) -> Toy8PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy8AgentsConfig:
        return self.model.agents

    @property
    def coordination(self) -> Toy8CoordinationConfig:
        return self.model.coordination


class Toy9EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_action_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    benefit: float = Field(default=1.2, ge=0.0)
    action_cost: float = Field(default=0.35, ge=0.0)
    payoff_ema_decay: float = Field(default=0.90, ge=0.0, lt=1.0)


class Toy9AgentGroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    fraction: float = Field(gt=0.0, le=1.0)
    local_rule: Literal["threshold", "payoff_learning"] = "threshold"
    initial_action_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.20, ge=0.0)
    coordination_enabled: bool = True


class Toy9PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal["heterogeneous_rules"] = "heterogeneous_rules"
    decision_mode: Literal["sampled", "argmax"] = "sampled"
    temperature: float = Field(default=1.0, gt=0.0)
    exploration_epsilon: float = Field(default=0.0, ge=0.0, le=1.0)


class Toy9AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, gt=1)
    init_mode: Literal["group_priors", "independent_init"] = "group_priors"
    groups: list[Toy9AgentGroupConfig] = Field(
        default_factory=lambda: [
            Toy9AgentGroupConfig(
                name="threshold",
                fraction=0.5,
                local_rule="threshold",
                threshold=0.40,
                coordination_enabled=True,
            ),
            Toy9AgentGroupConfig(
                name="payoff",
                fraction=0.5,
                local_rule="payoff_learning",
                coordination_enabled=False,
            ),
        ]
    )


class Toy9CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class Toy9GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["watts_strogatz"] = "watts_strogatz"
    k: int = Field(default=6, gt=0)
    rewire_probability: float = Field(default=0.1, ge=0.0, le=1.0)


class Toy9ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy9PolicyConfig = Field(default_factory=Toy9PolicyConfig)
    agents: Toy9AgentsConfig = Field(default_factory=Toy9AgentsConfig)
    coordination: Toy9CoordinationConfig = Field(default_factory=Toy9CoordinationConfig)


class Toy9DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy9"] = "toy9"
    environment: Toy9EnvironmentConfig = Field(default_factory=Toy9EnvironmentConfig)
    graph: Toy9GraphConfig = Field(default_factory=Toy9GraphConfig)


class Toy9Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy9ModelConfig = Field(default_factory=Toy9ModelConfig)
    domain: Toy9DomainConfig = Field(default_factory=Toy9DomainConfig)
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy9_config(self) -> "Toy9Config":
        if self.graph.k >= self.agents.count:
            raise ValueError("Toy 9 graph.k must be less than agents.count")
        if not self.agents.groups:
            raise ValueError("Toy 9 requires at least one agent group")
        if sum(group.fraction for group in self.agents.groups) <= 0.0:
            raise ValueError("Toy 9 group fractions must have positive mass")
        names = [group.name for group in self.agents.groups]
        if len(names) != len(set(names)):
            raise ValueError("Toy 9 agent group names must be unique")
        return self

    @property
    def environment(self) -> Toy9EnvironmentConfig:
        return self.domain.environment

    @property
    def graph(self) -> Toy9GraphConfig:
        return self.domain.graph

    @property
    def policy(self) -> Toy9PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy9AgentsConfig:
        return self.model.agents

    @property
    def coordination(self) -> Toy9CoordinationConfig:
        return self.model.coordination


class Toy10EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_initial: float = Field(default=85.0, ge=0.0)
    resource_carrying_capacity: float = Field(default=100.0, gt=0.0)
    resource_recovery_rate: float = Field(default=0.05, ge=0.0)
    extraction_scale: float = Field(default=7.0, ge=0.0)
    extraction_cost: float = Field(default=0.30, ge=0.0)
    base_price: float = Field(default=0.45, ge=0.0)
    demand_sensitivity: float = Field(default=0.50, ge=0.0)
    supply_sensitivity: float = Field(default=0.25, ge=0.0)
    initial_price_expectation_mean: float = Field(default=0.50, ge=0.0, le=1.0)
    initial_price_expectation_std: float = Field(default=0.08, ge=0.0)
    initial_conservation_norm_mean: float = Field(default=0.35, ge=0.0, le=1.0)
    initial_conservation_norm_std: float = Field(default=0.08, ge=0.0)

    @model_validator(mode="after")
    def validate_resource_initial(self) -> "Toy10EnvironmentConfig":
        if self.resource_initial > self.resource_carrying_capacity:
            raise ValueError(
                "Toy 10 resource_initial must be <= resource_carrying_capacity"
            )
        return self


class Toy10PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal["market_ecology"] = "market_ecology"
    learning_rate: float = Field(default=0.18, ge=0.0)
    revision_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    exploration_std: float = Field(default=0.02, ge=0.0)
    conservation_harvest_weight: float = Field(default=0.75, ge=0.0)
    social_harvest_gain: float = Field(default=1.0, ge=0.0)
    social_disagreement_penalty: float = Field(default=0.0, ge=0.0)
    reward_ema_decay: float = Field(default=0.90, ge=0.0, lt=1.0)


class Toy10AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, gt=1)
    init_mode: Literal["same_init", "independent_init"] = "independent_init"


class Toy10CoordinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixer: Literal["none", "output_average"] = "none"
    peer_rule: Literal["none", "output_similarity"] = "none"
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class Toy10NetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["watts_strogatz"] = "watts_strogatz"
    k: int = Field(default=6, gt=0)
    rewire_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    dynamic_rewire_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    candidate_pool_size: int = Field(default=8, gt=0)


class Toy10ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Toy10PolicyConfig = Field(default_factory=Toy10PolicyConfig)
    agents: Toy10AgentsConfig = Field(default_factory=Toy10AgentsConfig)
    coordination: Toy10CoordinationConfig = Field(
        default_factory=Toy10CoordinationConfig
    )


class Toy10DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toy: Literal["toy10"] = "toy10"
    environment: Toy10EnvironmentConfig = Field(default_factory=Toy10EnvironmentConfig)
    network: Toy10NetworkConfig = Field(default_factory=Toy10NetworkConfig)


class Toy10Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    simulation: SimulationConfig
    model: Toy10ModelConfig = Field(default_factory=Toy10ModelConfig)
    domain: Toy10DomainConfig = Field(default_factory=Toy10DomainConfig)
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_toy10_config(self) -> "Toy10Config":
        if self.network.k >= self.agents.count:
            raise ValueError("Toy 10 network.k must be less than agents.count")
        return self

    @property
    def environment(self) -> Toy10EnvironmentConfig:
        return self.domain.environment

    @property
    def network(self) -> Toy10NetworkConfig:
        return self.domain.network

    @property
    def graph(self) -> Toy10NetworkConfig:
        return self.domain.network

    @property
    def policy(self) -> Toy10PolicyConfig:
        return self.model.policy

    @property
    def agents(self) -> Toy10AgentsConfig:
        return self.model.agents

    @property
    def coordination(self) -> Toy10CoordinationConfig:
        return self.model.coordination


def load_toy1_config(path: str | Path) -> Toy1Config:
    """Load a Toy 1 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy1Config.model_validate(raw)


def load_toy2_config(path: str | Path) -> Toy2Config:
    """Load a Toy 2 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy2Config.model_validate(raw)


def load_toy3_config(path: str | Path) -> Toy3Config:
    """Load a Toy 3 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy3Config.model_validate(raw)


def load_toy4_config(path: str | Path) -> Toy4Config:
    """Load a Toy 4 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy4Config.model_validate(raw)


def load_toy5_config(path: str | Path) -> Toy5Config:
    """Load a Toy 5 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy5Config.model_validate(raw)


def load_toy6_config(path: str | Path) -> Toy6Config:
    """Load a Toy 6 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy6Config.model_validate(raw)


def load_toy7_config(path: str | Path) -> Toy7Config:
    """Load a Toy 7 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy7Config.model_validate(raw)


def load_toy8_config(path: str | Path) -> Toy8Config:
    """Load a Toy 8 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy8Config.model_validate(raw)


def load_toy9_config(path: str | Path) -> Toy9Config:
    """Load a Toy 9 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy9Config.model_validate(raw)


def load_toy10_config(path: str | Path) -> Toy10Config:
    """Load a Toy 10 YAML config and validate it."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Toy10Config.model_validate(raw)
