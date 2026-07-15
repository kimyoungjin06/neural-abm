"""Learning-agent PIVOT study through the torch-backed ``neural_abm.api``.

Study 2 of the researcher-pivot case study. Study 1 showed that hot-field
hype produces frequent but unproductive pivots. This study asks the question
a fixed-rule ABM cannot ask:

    Can researchers who learn from observed pivot outcomes self-correct the
    hype paradox, or does hype outpace learning?

All arms use the same sigmoid decision policy over the same nine observable
features; the only difference between arms is the learning rule, so any
divergence is attributable to it:

- ``frozen``: weights never update (the Study 1 push-pull rule in policy
  form). The classical-ABM control arm.
- ``imitative``: vicarious gradient learning from every observed neighbor
  pivot outcome (success and failure), via binary cross-entropy toward
  predicting productive pivots.
- ``cautionary``: negativity-biased social learning — the same update, but
  only failed pivots are treated as evidence. Asymmetric weighting of
  losses over gains is a documented regularity of human social learning.

Observed outcomes are survivorship-biased: only researchers who pivot reveal
an outcome. Under supportive environments early pivoters mostly succeed, so
imitative learners receive one-sided positive evidence — the conditions for
an information cascade. Whether that cascade materializes, and whether
cautionary learning instead yields targeted hype immunity, is what the
simulation measures. A weak L2 anchor toward the prior rule models
conservative belief updating.

Pivots are absorbing events: a researcher pivots once, the outcome
(productive or failed) is determined by structural fit at pivot time and
becomes visible to neighbors. Pivoted researchers keep broadcasting their
readiness on the social channel, so hype contagion and outcome learning
compete on the same network.

The simulation runs on the stable ``neural_abm.api`` lifecycle: agents
implement the NABM agent protocol, readiness mixes through a typed
bounded-scalar channel, and a commit adapter owns the domain transition.
Scenario definitions, population sampling, and the network builder are
imported from Study 1 so both studies share one environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from neural_abm.api import (
    BOUNDED_SCALAR_CHANNEL,
    CommitReport,
    NABMStep,
    NABMUnit,
    ObservationSpec,
    SocialBlock,
    SocialChannel,
    SocialMessageSpec,
    SocialMixResult,
    scalar_message_values,
    select_bounded_scalar_output_peers,
)

try:  # script execution (sys.path[0] == examples/)
    import research_pivot_study as study1
except ImportError:  # package-style import
    from examples import research_pivot_study as study1

FEATURES: tuple[str, ...] = (
    "field_opportunity",
    "resource_security",
    "network_support",
    "skill_distance",
    "reputation_risk",
    "openness",
    "funding_signal",
    "attention_signal",
    "peer_success_signal",
)

# Study 1 push-pull rule expressed as linear weights over FEATURES. The
# resource-insecurity term 0.12 * (1 - resource_security) becomes weight
# -0.12 plus +0.12 folded into the intercept.
FIXED_RULE_WEIGHTS: tuple[float, ...] = (
    0.28,  # field_opportunity
    -0.12,  # resource_security
    0.12,  # network_support
    -0.22,  # skill_distance
    -0.18,  # reputation_risk
    0.15,  # openness
    0.20,  # funding_signal
    0.25,  # attention_signal
    0.18,  # peer_success_signal
)
FIXED_RULE_INTERCEPT: float = 0.05 + 0.12
POLICY_LOGIT_SCALE: float = 4.0


@dataclass(frozen=True)
class LearningStudyConfig:
    agent_count: int = 120
    steps: int = 14
    burn_in_steps: int = 2
    replicates: int = 30
    base_seed: int = 20260715
    learning_rate: float = 1.5
    prior_anchor: float = 0.25
    arms: tuple[str, ...] = ("frozen", "imitative", "cautionary")
    local_alpha: float = 0.45
    local_noise_scale: float = 0.02
    social_alpha: float = 0.40
    peer_similarity_threshold: float = 0.68
    pivot_threshold: float = 0.34
    productive_threshold: float = 0.40


@dataclass
class PivotEvent:
    """A publicly observable pivot outcome."""

    agent_id: int
    features: np.ndarray
    productive: bool
    step: int


@dataclass
class PivotEnvironment:
    """Shared state the lifecycle callbacks read and write."""

    neighbors: list[list[int]]
    rng: np.random.Generator
    config: LearningStudyConfig
    step: int = 0
    last_events: list[PivotEvent] = field(default_factory=list)
    new_events: list[PivotEvent] = field(default_factory=list)

    def advance(self) -> None:
        self.step += 1
        self.last_events = self.new_events
        self.new_events = []


def _feature_vector(researcher: study1.Researcher) -> np.ndarray:
    return np.asarray(
        [float(getattr(researcher, name)) for name in FEATURES],
        dtype=np.float64,
    )


def _initial_policy() -> tuple[torch.Tensor, torch.Tensor]:
    weight = POLICY_LOGIT_SCALE * torch.tensor(
        FIXED_RULE_WEIGHTS,
        dtype=torch.float64,
    )
    bias = torch.tensor(
        POLICY_LOGIT_SCALE * (FIXED_RULE_INTERCEPT - 0.5),
        dtype=torch.float64,
    )
    return weight.requires_grad_(True), bias.requires_grad_(True)


@dataclass
class PivotPolicyAgent:
    """NABM-protocol agent whose readiness target is a learnable policy."""

    agent_id: int
    base: study1.Researcher
    environment: PivotEnvironment
    learning_rate: float
    learning_mode: str = "imitative"
    prior_anchor: float = 0.25
    weight: torch.Tensor = field(default_factory=lambda: _initial_policy()[0])
    bias: torch.Tensor = field(default_factory=lambda: _initial_policy()[1])
    features: np.ndarray = field(init=False)
    pivot_step: int | None = None
    last_policy_output: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.features = _feature_vector(self.base)
        self.prior_weight = self.weight.detach().clone()
        self.prior_bias = self.bias.detach().clone()
        self.last_policy_output = self.policy_output(self.features)

    # -- policy ---------------------------------------------------------

    def policy_logit(self, features: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(features, dtype=torch.float64)
        return x @ self.weight + self.bias

    def policy_output(self, features: np.ndarray) -> float:
        with torch.no_grad():
            return float(torch.sigmoid(self.policy_logit(features)))

    def attention_weight(self) -> float:
        with torch.no_grad():
            return float(self.weight[FEATURES.index("attention_signal")])

    def _learn_from_events(self, events: Sequence[PivotEvent]) -> None:
        if self.learning_rate <= 0.0:
            return
        neighbor_ids = set(self.environment.neighbors[self.agent_id])
        neighbor_ids.add(self.agent_id)
        samples = [event for event in events if event.agent_id in neighbor_ids]
        if self.learning_mode == "cautionary":
            samples = [event for event in samples if not event.productive]
        if not samples:
            return
        inputs = torch.as_tensor(
            np.stack([event.features for event in samples]),
            dtype=torch.float64,
        )
        targets = torch.as_tensor(
            [1.0 if event.productive else 0.0 for event in samples],
            dtype=torch.float64,
        )
        logits = inputs @ self.weight + self.bias
        # Conservative belief updating: observed outcomes pull the policy,
        # the prior rule anchors it. Without the anchor, one-sided success
        # evidence (e.g. under seed grants) triggers an unbounded pivot
        # cascade through the bias term.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
        ) + 0.5 * self.prior_anchor * (
            torch.sum((self.weight - self.prior_weight) ** 2)
            + (self.bias - self.prior_bias) ** 2
        )
        loss.backward()
        with torch.no_grad():
            self.weight -= self.learning_rate * self.weight.grad
            self.bias -= self.learning_rate * self.bias.grad
        self.weight.grad = None
        self.bias.grad = None

    # -- NABM agent protocol --------------------------------------------

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="pivot_features",
            tensor_shape=(len(FEATURES),),
            dtype=torch.float64,
            description="observable researcher profile",
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "pivot_readiness",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("latent_summary",),
            probability_keys=("pivot_readiness",),
        )

    def observe(self, x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().to(dtype=torch.float64)
        return torch.as_tensor(self.features, dtype=torch.float64)

    def act_or_predict(self, observation: Any) -> torch.Tensor:
        observed = self.observe(observation)
        with torch.no_grad():
            return torch.sigmoid(observed @ self.weight + self.bias).reshape(1)

    def local_update(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        self._learn_from_events(self.environment.last_events)
        self.last_policy_output = self.policy_output(self.features)
        if self.base.pivoted:
            return 0.0
        config = self.environment.config
        before = float(self.base.pivot_readiness)
        noise = float(self.environment.rng.normal(0.0, config.local_noise_scale))
        self.base.pivot_readiness = float(
            np.clip(
                (1.0 - config.local_alpha) * before
                + config.local_alpha * self.last_policy_output
                + noise,
                0.0,
                1.0,
            )
        )
        return abs(self.base.pivot_readiness - before)

    def social_message(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "agent_id": self.agent_id,
            "pivot_readiness": float(self.base.pivot_readiness),
            "latent_summary": torch.tensor(
                [self.base.pivot_readiness, self.last_policy_output],
                dtype=torch.float32,
            ),
            "confidence": 1.0,
            "param_norm": float(torch.linalg.vector_norm(self.weight.detach())),
        }

    def log_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "agent_id": self.agent_id,
            "career_stage": self.base.career_stage,
            "pivot_readiness": float(self.base.pivot_readiness),
            "policy_output": self.last_policy_output,
            "attention_weight": self.attention_weight(),
            "pivoted": self.base.pivoted,
            "productive_pivot": self.base.productive_pivot,
            "pivot_step": self.pivot_step,
        }


@dataclass
class PivotCommitAdapter:
    """Domain-owned transition: commit mixed readiness, absorb pivot events."""

    agents: Sequence[PivotPolicyAgent]
    environment: PivotEnvironment

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        config = self.environment.config
        values = np.asarray(mix_result.mixed_values, dtype=np.float64)
        committed: list[int] = []
        losses = [0.0 for _agent in self.agents]
        # Threshold transitions are disabled during burn-in so absorbing
        # pivots reflect settled dynamics, not initial-condition transients.
        transitions_open = self.environment.step > config.burn_in_steps
        for index, agent in enumerate(self.agents):
            if agent.base.pivoted:
                continue
            before = float(agent.base.pivot_readiness)
            after = float(np.clip(values[index], 0.0, 1.0))
            agent.base.pivot_readiness = after
            losses[index] = abs(after - before)
            committed.append(agent.agent_id)
            if transitions_open and after >= config.pivot_threshold:
                agent.base.pivoted = True
                agent.base.productive_pivot = (
                    agent.base.productive_fit >= config.productive_threshold
                )
                agent.pivot_step = self.environment.step
                self.environment.new_events.append(
                    PivotEvent(
                        agent_id=agent.agent_id,
                        features=agent.features.copy(),
                        productive=agent.base.productive_pivot,
                        step=self.environment.step,
                    )
                )
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


def _make_peer_selector(
    environment: PivotEnvironment,
    config: LearningStudyConfig,
) -> Any:
    def select(messages: Sequence[Mapping[str, Any]]) -> list[list[int]]:
        values = np.asarray(
            [float(message["pivot_readiness"]) for message in messages],
            dtype=np.float64,
        )
        return select_bounded_scalar_output_peers(
            neighbors=environment.neighbors,
            values=values,
            peer_rule="output_similarity",
            threshold=config.peer_similarity_threshold,
            lower_bound=0.0,
            upper_bound=1.0,
        ).peer_ids

    return select


def run_arm(
    *,
    scenario: Any,
    config: LearningStudyConfig,
    replicate: int,
    learning_mode: str,
) -> dict[str, Any]:
    """Run one scenario x policy arm for one replicate."""

    rng = np.random.default_rng(np.random.SeedSequence((config.base_seed, replicate)))
    study_config = study1.StudyConfig(
        agent_count=config.agent_count,
        local_alpha=config.local_alpha,
        local_noise_scale=config.local_noise_scale,
        social_alpha=config.social_alpha,
        peer_similarity_threshold=config.peer_similarity_threshold,
        pivot_threshold=config.pivot_threshold,
        productive_threshold=config.productive_threshold,
    )
    context = SimpleNamespace(scenario=scenario, replicate=replicate, rng=rng)
    researchers = study1.make_researchers(context, study_config)
    neighbors = study1.build_stage_assortative_network(
        researchers,
        context,
        study_config,
    )
    environment = PivotEnvironment(neighbors=neighbors, rng=rng, config=config)
    agents = [
        PivotPolicyAgent(
            agent_id=researcher.agent_id,
            base=researcher,
            environment=environment,
            learning_rate=(0.0 if learning_mode == "frozen" else config.learning_rate),
            learning_mode=learning_mode,
            prior_anchor=config.prior_anchor,
        )
        for researcher in researchers
    ]
    unit = NABMUnit(
        agents=agents,
        step=NABMStep(
            social_block=SocialBlock(alpha=config.social_alpha),
            channel=SocialChannel(
                name="pivot_readiness",
                kind=BOUNDED_SCALAR_CHANNEL,
                commit_mode="domain_pivot_threshold",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            commit_adapter=PivotCommitAdapter(agents=agents, environment=environment),
        ),
        peer_selector=_make_peer_selector(environment, config),
        social_value_builder=scalar_message_values("pivot_readiness"),
    )

    count = len(agents)
    cumulative_failed: list[float] = []
    cumulative_productive: list[float] = []
    new_pivots_per_step: list[int] = []
    new_failures_per_step: list[int] = []
    mean_attention_weight: list[float] = []
    for _step in range(config.steps):
        environment.advance()
        unit.run(collect_logs=False)
        new_pivots_per_step.append(len(environment.new_events))
        new_failures_per_step.append(
            sum(1 for event in environment.new_events if not event.productive)
        )
        failed = sum(
            1
            for agent in agents
            if agent.base.pivoted and not agent.base.productive_pivot
        )
        productive = sum(1 for agent in agents if agent.base.productive_pivot)
        cumulative_failed.append(failed / count)
        cumulative_productive.append(productive / count)
        mean_attention_weight.append(
            float(np.mean([agent.attention_weight() for agent in agents]))
        )

    pivots = sum(1 for agent in agents if agent.base.pivoted)
    productive = sum(1 for agent in agents if agent.base.productive_pivot)
    return {
        "pivot_rate": pivots / count,
        "productive_pivot_rate": productive / count,
        "failed_pivot_rate": (pivots - productive) / count,
        "productive_share_of_pivots": productive / pivots if pivots else 0.0,
        "cumulative_failed_rate": cumulative_failed,
        "cumulative_productive_rate": cumulative_productive,
        "new_pivots_per_step": new_pivots_per_step,
        "new_failures_per_step": new_failures_per_step,
        "mean_attention_weight": mean_attention_weight,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    low, high = (float(v) for v in np.percentile(array, [2.5, 97.5]))
    return {
        "mean": round(float(array.mean()), 4),
        "std": round(std, 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
    }


def run_learning_study(config: LearningStudyConfig) -> dict[str, Any]:
    scenario_names = ("baseline", "interdisciplinary_seed_grants", "hot_field_hype")
    scenarios = [
        scenario
        for scenario in study1.study_scenarios()
        if scenario.name in scenario_names
    ]
    arms: dict[str, dict[str, list[dict[str, Any]]]] = {
        scenario.name: {arm: [] for arm in config.arms} for scenario in scenarios
    }
    for replicate in range(config.replicates):
        for scenario in scenarios:
            for arm_name in config.arms:
                arms[scenario.name][arm_name].append(
                    run_arm(
                        scenario=scenario,
                        config=config,
                        replicate=replicate,
                        learning_mode=arm_name,
                    )
                )

    scenario_payload: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    for name, by_arm in arms.items():
        scenario_payload[name] = {}
        for arm_name, runs in by_arm.items():
            trajectories = np.asarray([run["cumulative_failed_rate"] for run in runs])
            productive_trajectories = np.asarray(
                [run["cumulative_productive_rate"] for run in runs]
            )
            attention = np.asarray([run["mean_attention_weight"] for run in runs])
            scenario_payload[name][arm_name] = {
                "pivot_rate": _distribution([run["pivot_rate"] for run in runs]),
                "productive_pivot_rate": _distribution(
                    [run["productive_pivot_rate"] for run in runs]
                ),
                "failed_pivot_rate": _distribution(
                    [run["failed_pivot_rate"] for run in runs]
                ),
                "productive_share_of_pivots": _distribution(
                    [run["productive_share_of_pivots"] for run in runs]
                ),
                "cumulative_failed_rate_mean": [
                    round(float(v), 4) for v in trajectories.mean(axis=0)
                ],
                "cumulative_failed_rate_std": [
                    round(float(v), 4) for v in trajectories.std(axis=0)
                ],
                "cumulative_productive_rate_mean": [
                    round(float(v), 4) for v in productive_trajectories.mean(axis=0)
                ],
                "mean_attention_weight": [
                    round(float(v), 4) for v in attention.mean(axis=0)
                ],
            }
        for arm_name in config.arms:
            if arm_name == "frozen":
                continue
            for outcome in ("failed_pivot_rate", "productive_pivot_rate"):
                deltas = [
                    learning_run[outcome] - frozen_run[outcome]
                    for frozen_run, learning_run in zip(
                        by_arm["frozen"],
                        by_arm[arm_name],
                        strict=True,
                    )
                ]
                comparisons.append(
                    {
                        "scenario": name,
                        "outcome_field": outcome,
                        "arm": f"{arm_name} - frozen",
                        "replicates": config.replicates,
                        **_distribution(deltas),
                    }
                )

    return {
        "status": "ok",
        "surface": "neural_abm.api",
        "default_profile": "torch-backed",
        "torch_loaded": "torch" in sys.modules,
        "research_question": ("can_outcome_learning_self_correct_the_hype_paradox"),
        "config": {
            "agent_count": config.agent_count,
            "steps": config.steps,
            "replicates": config.replicates,
            "base_seed": config.base_seed,
            "learning_rate": config.learning_rate,
            "pivot_threshold": config.pivot_threshold,
            "productive_threshold": config.productive_threshold,
        },
        "features": list(FEATURES),
        "scenarios": scenario_payload,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.quick:
        overrides.update({"agent_count": 40, "steps": 6, "replicates": 4})
    if args.agents is not None:
        overrides["agent_count"] = args.agents
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.replicates is not None:
        overrides["replicates"] = args.replicates
    if args.base_seed is not None:
        overrides["base_seed"] = args.base_seed
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    config = LearningStudyConfig(**overrides)

    payload = run_learning_study(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(args.output),
                    "comparisons": payload["comparisons"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
