"""Reusable Neural ABM unit lifecycle and commit adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from neural_abm.social import SocialBlock, SocialChannel, SocialMixResult
from neural_abm.unit_core import (
    CommitAdapter,
    CommitReport,
    LocalUpdateAdapter,
    LocalUpdateReport,
    NABMLocalStep,
    NABMStepResult,
    PeerSelector,
    SocialDiagnostics,
    SocialValueBuilder,
    social_diagnostics,
)

__all__ = [
    "CommitAdapter",
    "CommitReport",
    "DistributionDistillationAdapter",
    "LocalUpdateAdapter",
    "LocalUpdateReport",
    "NABMAgent",
    "NABMLocalStep",
    "NABMStep",
    "NABMStepResult",
    "NABMUnit",
    "NABMUnitReport",
    "ObservationSpec",
    "PeerSelector",
    "SocialDiagnostics",
    "SocialMessageSpec",
    "SocialValueBuilder",
    "StateDictLoadAdapter",
    "TensorDistillationAdapter",
    "scalar_message_values",
    "social_diagnostics",
    "state_dict_values",
    "tensor_message_values",
]


@dataclass(frozen=True)
class ObservationSpec:
    """Tensor observation contract exposed by a NABM-compatible agent."""

    name: str
    tensor_shape: tuple[int | None, ...] | None = None
    dtype: torch.dtype | None = torch.float32
    description: str = ""

    def validate(self, observation: Any) -> None:
        if self.tensor_shape is None and self.dtype is None:
            return
        if not torch.is_tensor(observation):
            raise ValueError(f"{self.name} observation must be a torch.Tensor")
        if self.dtype is not None and observation.dtype != self.dtype:
            raise ValueError(
                f"{self.name} observation dtype must be {self.dtype}, "
                f"got {observation.dtype}"
            )
        if self.tensor_shape is None:
            return

        observed_shape = tuple(int(dim) for dim in observation.shape)
        expected_shape = self.tensor_shape
        if (
            expected_shape
            and expected_shape[0] is None
            and len(observed_shape) == len(expected_shape) - 1
        ):
            expected_shape = expected_shape[1:]
        if len(observed_shape) != len(expected_shape):
            raise ValueError(
                f"{self.name} observation rank must be {len(expected_shape)}, "
                f"got {len(observed_shape)}"
            )
        for dim_id, (observed, expected) in enumerate(
            zip(observed_shape, expected_shape, strict=True)
        ):
            if expected is not None and observed != expected:
                raise ValueError(
                    f"{self.name} observation dim {dim_id} must be {expected}, "
                    f"got {observed}"
                )


@dataclass(frozen=True)
class SocialMessageSpec:
    """Schema for bounded social messages emitted by NABM-compatible agents."""

    required_keys: tuple[str, ...] = (
        "agent_id",
        "latent_summary",
        "confidence",
        "param_norm",
    )
    tensor_keys: tuple[str, ...] = ("latent_summary",)
    probability_keys: tuple[str, ...] = ()

    def validate(self, message: Mapping[str, Any]) -> None:
        for key in self.required_keys:
            if key not in message:
                raise ValueError(f"social message missing required key: {key}")
        for key in self.tensor_keys:
            value = message[key]
            if not torch.is_tensor(value):
                raise ValueError(f"social message {key!r} must be a torch.Tensor")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"social message {key!r} must contain finite values")
        for key in self.probability_keys:
            value = message[key]
            tensor = (
                value.detach()
                if torch.is_tensor(value)
                else torch.as_tensor(value, dtype=torch.float32)
            )
            if not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"social message {key!r} must contain finite values")
            if bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
                raise ValueError(f"social message {key!r} must lie in [0, 1]")
        confidence = float(message["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("social message confidence must lie in [0, 1]")
        if float(message["param_norm"]) < 0.0:
            raise ValueError("social message param_norm must be non-negative")


@runtime_checkable
class NABMAgent(Protocol):
    """Minimal protocol expected by reusable NABM lifecycle components."""

    agent_id: int

    def observation_spec(self) -> ObservationSpec:
        """Return the observation schema accepted by this agent."""

    def social_message_spec(self) -> SocialMessageSpec:
        """Return the social-message schema emitted by this agent."""

    def observe(self, x: Any) -> Any:
        """Convert environment input into the agent observation."""

    def act_or_predict(self, observation: Any) -> Any:
        """Emit an action or predictive output for an observation."""

    def local_update(self, *args: Any, **kwargs: Any) -> float:
        """Run one local learning/update phase."""

    def social_message(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Emit bounded state used by social selection or logging."""

    def log_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Emit flat diagnostics for logging."""


@dataclass
class StateDictLoadAdapter:
    """Commit state-dict channels by loading mixed states into agent models."""

    agents: Sequence[Any]
    skip_empty_peers: bool = True

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        committed: list[int] = []
        for agent_id, agent in enumerate(self.agents):
            if self.skip_empty_peers and not mix_result.peer_ids[agent_id]:
                continue
            agent.model.load_state_dict(mix_result.mixed_values[agent_id])
            committed.append(int(getattr(agent, "agent_id", agent_id)))
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
        )


@dataclass
class DistributionDistillationAdapter:
    """Commit distribution targets by a supervised distillation step."""

    agents: Sequence[Any]
    logits_fn: Callable[[Any, int], torch.Tensor]
    optimizer_fn: Callable[[Any, int], torch.optim.Optimizer]
    loss_mode: str = "kl"
    skip_empty_peers: bool = True

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        losses = [0.0 for _ in self.agents]
        committed: list[int] = []
        for agent_id, agent in enumerate(self.agents):
            if self.skip_empty_peers and not mix_result.peer_ids[agent_id]:
                continue
            target = mix_result.mixed_values[agent_id].detach()
            logits = self.logits_fn(agent, agent_id)
            if logits.ndim == target.ndim + 1 and logits.shape[0] == 1:
                target = target.unsqueeze(0)
            if self.loss_mode == "kl":
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                loss = nn.functional.kl_div(log_probs, target, reduction="batchmean")
            elif self.loss_mode == "cross_entropy":
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                loss = -(target * log_probs).sum()
            else:
                raise ValueError(f"Unsupported distribution loss mode: {self.loss_mode}")
            optimizer = self.optimizer_fn(agent, agent_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[agent_id] = float(loss.detach().cpu())
            committed.append(int(getattr(agent, "agent_id", agent_id)))
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


@dataclass
class TensorDistillationAdapter:
    """Commit tensor targets with an MSE distillation step."""

    agents: Sequence[Any]
    tensor_fn: Callable[[Any, int], torch.Tensor]
    optimizer_fn: Callable[[Any, int], torch.optim.Optimizer]
    skip_empty_peers: bool = True

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        losses = [0.0 for _ in self.agents]
        committed: list[int] = []
        for agent_id, agent in enumerate(self.agents):
            if self.skip_empty_peers and not mix_result.peer_ids[agent_id]:
                continue
            target = mix_result.mixed_values[agent_id].detach()
            value = self.tensor_fn(agent, agent_id)
            loss = nn.functional.mse_loss(value, target)
            optimizer = self.optimizer_fn(agent, agent_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[agent_id] = float(loss.detach().cpu())
            committed.append(int(getattr(agent, "agent_id", agent_id)))
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


@dataclass
class NABMStep:
    """Reusable lifecycle unit for social mix and optional commit."""

    social_block: SocialBlock
    channel: SocialChannel
    commit_adapter: CommitAdapter | None = None

    def mix(self, values: Any, peer_ids: list[list[int]]) -> SocialMixResult:
        return self.social_block.mix(
            channel=self.channel,
            values=values,
            peer_ids=peer_ids,
        )

    def run(self, values: Any, peer_ids: list[list[int]]) -> NABMStepResult:
        mix_result = self.mix(values=values, peer_ids=peer_ids)
        if self.commit_adapter is None:
            commit_report = CommitReport.from_mix_result(mix_result)
        else:
            commit_report = self.commit_adapter.commit(mix_result)
        diagnostics = social_diagnostics(
            SocialMixResult(
                mixed_values=mix_result.mixed_values,
                losses=commit_report.losses,
                update_norms=mix_result.update_norms,
                peer_ids=mix_result.peer_ids,
                channel=mix_result.channel,
                commit_mode=mix_result.commit_mode,
            )
        )
        return NABMStepResult(
            mix=mix_result,
            commit=commit_report,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class NABMUnitReport:
    """Result of one generic local/social/logging NABM unit step."""

    local_losses: list[float]
    social_messages: list[dict[str, Any]]
    peer_ids: list[list[int]]
    social_step: NABMStepResult
    agent_logs: list[dict[str, Any]]

    @property
    def mean_local_loss(self) -> float:
        return (
            float(sum(self.local_losses) / len(self.local_losses))
            if self.local_losses
            else 0.0
        )

    def aggregate_row(self) -> dict[str, Any]:
        row = {
            "agent_count": len(self.social_messages),
            "mean_local_loss": self.mean_local_loss,
        }
        row.update(self.social_step.diagnostics.aggregate_row())
        return row

    def micro_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for agent_index, log_row in enumerate(self.agent_logs):
            row = {
                "agent_id": int(log_row.get("agent_id", agent_index)),
                "local_loss": (
                    self.local_losses[agent_index]
                    if agent_index < len(self.local_losses)
                    else 0.0
                ),
            }
            row.update(log_row)
            row.update(self.social_step.diagnostics.micro_row(agent_index))
            rows.append(row)
        return rows


def scalar_message_values(field: str) -> SocialValueBuilder:
    """Build a scalar-valued social channel from validated message fields."""

    def build(
        agents: Sequence[NABMAgent],
        messages: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        del agents
        return np.asarray([float(message[field]) for message in messages], dtype=float)

    return build


def tensor_message_values(field: str) -> SocialValueBuilder:
    """Build a stacked tensor social channel from validated message fields."""

    def build(
        agents: Sequence[NABMAgent],
        messages: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        del agents
        values = []
        for message in messages:
            value = message[field]
            if not torch.is_tensor(value):
                raise ValueError(f"social message field {field!r} must be a tensor")
            values.append(value.detach())
        if not values:
            return torch.empty(0)
        return torch.stack(values, dim=0)

    return build


def state_dict_values() -> SocialValueBuilder:
    """Build a parameter-state social channel from agent model state dicts."""

    def build(
        agents: Sequence[NABMAgent],
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        del messages
        states: list[dict[str, torch.Tensor]] = []
        for agent in agents:
            model = getattr(agent, "model", None)
            if model is None or not hasattr(model, "state_dict"):
                raise ValueError("state_dict_values requires agents with model.state_dict")
            states.append(
                {key: value.detach().clone() for key, value in model.state_dict().items()}
            )
        return states

    return build


@dataclass
class NABMUnit:
    """Generic NABM unit lifecycle over agents, local updates, and social mix."""

    agents: Sequence[NABMAgent]
    step: NABMStep
    peer_selector: PeerSelector
    social_value_builder: SocialValueBuilder

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("NABMUnit requires at least one agent")

    def local_update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[float]:
        return [float(agent.local_update(*args, **kwargs)) for agent in self.agents]

    def social_messages(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for agent in self.agents:
            message = agent.social_message(*args, **kwargs)
            agent.social_message_spec().validate(message)
            messages.append(message)
        return messages

    def log_states(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [agent.log_state(*args, **kwargs) for agent in self.agents]

    def run(
        self,
        *,
        local_update_args: Sequence[Any] = (),
        local_update_kwargs: Mapping[str, Any] | None = None,
        message_args: Sequence[Any] = (),
        message_kwargs: Mapping[str, Any] | None = None,
        log_args: Sequence[Any] = (),
        log_kwargs: Mapping[str, Any] | None = None,
        run_local_update: bool = True,
        collect_logs: bool = True,
    ) -> NABMUnitReport:
        local_kwargs = dict(local_update_kwargs or {})
        message_call_kwargs = dict(message_kwargs or {})
        log_call_kwargs = dict(log_kwargs or {})
        local_losses = (
            self.local_update(*local_update_args, **local_kwargs)
            if run_local_update
            else [0.0 for _ in self.agents]
        )
        messages = self.social_messages(*message_args, **message_call_kwargs)
        peer_ids = self.peer_selector(messages)
        values = self.social_value_builder(self.agents, messages)
        social_step = self.step.run(values=values, peer_ids=peer_ids)
        agent_logs = (
            self.log_states(*log_args, **log_call_kwargs)
            if collect_logs
            else [
                {"agent_id": int(getattr(agent, "agent_id", agent_index))}
                for agent_index, agent in enumerate(self.agents)
            ]
        )
        return NABMUnitReport(
            local_losses=local_losses,
            social_messages=messages,
            peer_ids=peer_ids,
            social_step=social_step,
            agent_logs=agent_logs,
        )
