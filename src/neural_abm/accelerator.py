"""Torch accelerator helpers and batched neural policy kernels."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from neural_abm.losses import LossVector, TensorBackedLossVector

NeuralUpdateBackend = Literal["loop", "batched", "tensor_batched"]
NeuralUpdateBackendRequest = Literal["loop", "batched", "tensor_batched", "auto"]
AUTO_BATCHED_TRAINING_MIN_AGENTS = 256
AcceleratorTimingRecorder = Callable[[str, float], None]
AcceleratorTimingSynchronizer = Callable[[], None]


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_torch_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a user-facing torch device string.

    Explicit accelerator requests fail when unavailable. The special
    ``"auto"``/``"gpu"`` values select CUDA, then MPS, then CPU.
    """

    if requested is None:
        return torch.device("cpu")

    raw = str(requested).strip().lower()
    if not raw or raw == "cpu":
        return torch.device("cpu")

    if raw in {"auto", "accelerator", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested torch device {raw!r}, but CUDA is not available"
        )
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError(f"Requested torch device {raw!r}, but MPS is not available")
    return device


def is_accelerator_device(device: torch.device | str) -> bool:
    return torch.device(device).type in {"cuda", "mps", "xpu"}


def _supports_batched_adam_flat_update(device: torch.device | str) -> bool:
    # The flattened Adam kernel has CUDA coverage; other accelerators stay on
    # the shape-preserving grouped path until they have explicit validation.
    return torch.device(device).type == "cuda"


def resolve_neural_update_backend(
    requested: NeuralUpdateBackendRequest | str,
    *,
    device: torch.device | str,
    agent_count: int,
) -> NeuralUpdateBackend:
    """Resolve loop/batched neural training backend requests."""

    raw = str(requested).strip().lower()
    if raw in {"loop", "batched", "tensor_batched"}:
        return raw  # type: ignore[return-value]
    if raw != "auto":
        raise ValueError(
            "neural_update_backend must be 'loop', 'batched', "
            "'tensor_batched', or 'auto'"
        )
    if is_accelerator_device(device) or agent_count >= AUTO_BATCHED_TRAINING_MIN_AGENTS:
        return "batched"
    return "loop"


@dataclass(frozen=True)
class BatchedMLPParameters:
    """Stacked parameters for per-agent one-hidden-layer MLP policies."""

    fc1_weight: torch.Tensor
    fc1_bias: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_bias: torch.Tensor

    @classmethod
    def from_models(
        cls,
        models: Sequence[nn.Module],
        *,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> "BatchedMLPParameters":
        if not models:
            raise ValueError("At least one model is required")

        first = models[0]
        _validate_batched_mlp_model(first)
        resolved_device = first.fc1.weight.device if device is None else torch.device(device)
        input_dim = first.fc1.in_features
        hidden_dim = first.fc1.out_features
        output_dim = first.fc2.out_features

        for index, model in enumerate(models[1:], start=1):
            _validate_batched_mlp_model(model)
            if (
                model.fc1.in_features != input_dim
                or model.fc1.out_features != hidden_dim
                or model.fc2.in_features != hidden_dim
                or model.fc2.out_features != output_dim
            ):
                raise ValueError(
                    "All batched MLP models must share fc1/fc2 dimensions; "
                    f"model {index} differs"
                )

        return cls(
            fc1_weight=torch.stack(
                [
                    _parameter_snapshot(
                        model.fc1.weight,
                        resolved_device,
                        requires_grad=requires_grad,
                    )
                    for model in models
                ],
                dim=0,
            ),
            fc1_bias=torch.stack(
                [
                    _parameter_snapshot(
                        model.fc1.bias,
                        resolved_device,
                        requires_grad=requires_grad,
                    )
                    for model in models
                ],
                dim=0,
            ),
            fc2_weight=torch.stack(
                [
                    _parameter_snapshot(
                        model.fc2.weight,
                        resolved_device,
                        requires_grad=requires_grad,
                    )
                    for model in models
                ],
                dim=0,
            ),
            fc2_bias=torch.stack(
                [
                    _parameter_snapshot(
                        model.fc2.bias,
                        resolved_device,
                        requires_grad=requires_grad,
                    )
                    for model in models
                ],
                dim=0,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.fc1_weight.device

    @property
    def agent_count(self) -> int:
        return int(self.fc1_weight.shape[0])

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError("Batched MLP observations must have shape [agents, input]")
        if observations.shape[0] != self.agent_count:
            raise ValueError(
                "Observation count must match model count; "
                f"got {observations.shape[0]} observations for {self.agent_count} models"
            )
        observations = observations.to(
            device=self.device,
            dtype=self.fc1_weight.dtype,
        )
        hidden = torch.einsum("ni,nhi->nh", observations, self.fc1_weight)
        hidden = F.relu(hidden + self.fc1_bias)
        return (
            torch.einsum("nh,noh->no", hidden, self.fc2_weight)
            + self.fc2_bias
        )

    def probabilities(
        self,
        observations: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        return torch.softmax(self.logits(observations) / temperature, dim=-1)

    def trainable_clone(self) -> "BatchedMLPParameters":
        """Return a detached, grad-enabled clone on the current device."""

        return BatchedMLPParameters(
            fc1_weight=self.fc1_weight.detach().clone().requires_grad_(True),
            fc1_bias=self.fc1_bias.detach().clone().requires_grad_(True),
            fc2_weight=self.fc2_weight.detach().clone().requires_grad_(True),
            fc2_bias=self.fc2_bias.detach().clone().requires_grad_(True),
        )

    def trainable_view(self) -> "BatchedMLPParameters":
        """Return a detached, grad-enabled view sharing current tensor storage."""

        return BatchedMLPParameters(
            fc1_weight=self.fc1_weight.detach().requires_grad_(True),
            fc1_bias=self.fc1_bias.detach().requires_grad_(True),
            fc2_weight=self.fc2_weight.detach().requires_grad_(True),
            fc2_bias=self.fc2_bias.detach().requires_grad_(True),
        )

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
        )

    def detached_clone(self) -> "BatchedMLPParameters":
        """Return a detached inference snapshot on the current device."""

        return BatchedMLPParameters(
            fc1_weight=self.fc1_weight.detach().clone(),
            fc1_bias=self.fc1_bias.detach().clone(),
            fc2_weight=self.fc2_weight.detach().clone(),
            fc2_bias=self.fc2_bias.detach().clone(),
        )

    def detached(self) -> "BatchedMLPParameters":
        """Return a detached inference view on the current device."""

        return BatchedMLPParameters(
            fc1_weight=self.fc1_weight.detach(),
            fc1_bias=self.fc1_bias.detach(),
            fc2_weight=self.fc2_weight.detach(),
            fc2_bias=self.fc2_bias.detach(),
        )


@dataclass(frozen=True)
class BatchedMLPUpdateResult:
    losses: LossVector
    updated_parameters: BatchedMLPParameters | None
    used_batched_optimizer: bool


@dataclass
class BatchedAdamStateCache:
    """Reusable batched Adam state for same-shaped per-agent MLP optimizers."""

    exp_avg: BatchedMLPParameters
    exp_avg_sq: BatchedMLPParameters
    steps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    models: tuple[nn.Module, ...]
    optimizers: tuple[torch.optim.Adam, ...]
    parameter_ref_groups: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]
    state_groups: tuple[
        tuple[dict[str, torch.Tensor], ...],
        tuple[dict[str, torch.Tensor], ...],
        tuple[dict[str, torch.Tensor], ...],
        tuple[dict[str, torch.Tensor], ...],
    ]

    @classmethod
    def from_agents(
        cls,
        agents: Sequence[object],
        *,
        device: torch.device | str | None = None,
    ) -> "BatchedAdamStateCache":
        models: list[nn.Module] = []
        optimizers: list[torch.optim.Adam] = []
        for agent_id, agent in enumerate(agents):
            model, optimizer = _model_and_optimizer_from_agent(agent, agent_id)
            if not isinstance(optimizer, torch.optim.Adam):
                raise ValueError("Batched Adam state cache requires Adam optimizers")
            if not _adam_optimizer_is_supported(optimizer, model):
                raise ValueError("Unsupported Adam optimizer configuration")
            models.append(model)
            optimizers.append(optimizer)
        if not models:
            raise ValueError("At least one agent is required")

        reference_group = optimizers[0].param_groups[0]
        for optimizer in optimizers[1:]:
            if not _adam_groups_match(reference_group, optimizer.param_groups[0]):
                raise ValueError("All cached Adam optimizers must share hyperparameters")

        resolved_device = (
            models[0].fc1.weight.device
            if device is None
            else torch.device(device)
        )
        parameter_ref_groups = _batched_mlp_parameter_refs(models)
        state_groups: list[list[dict[str, torch.Tensor]]] = []
        for refs in parameter_ref_groups:
            states = []
            for optimizer, parameter in zip(optimizers, refs, strict=True):
                state = _adam_state_for_parameter(optimizer, parameter)
                if state is None:
                    raise ValueError("Unsupported Adam optimizer state")
                states.append(state)
            state_groups.append(states)

        state_tensors = [
            (
                _stack_adam_state_group(
                    states,
                    "exp_avg",
                    device=resolved_device,
                    dtype=refs[0].dtype,
                ),
                _stack_adam_state_group(
                    states,
                    "exp_avg_sq",
                    device=resolved_device,
                    dtype=refs[0].dtype,
                ),
                _stack_adam_state_group(
                    states,
                    "step",
                    device=resolved_device,
                    dtype=refs[0].dtype,
                ),
            )
            for states, refs in zip(
                state_groups,
                parameter_ref_groups,
                strict=True,
            )
        ]
        return cls(
            exp_avg=BatchedMLPParameters(
                fc1_weight=state_tensors[0][0],
                fc1_bias=state_tensors[1][0],
                fc2_weight=state_tensors[2][0],
                fc2_bias=state_tensors[3][0],
            ),
            exp_avg_sq=BatchedMLPParameters(
                fc1_weight=state_tensors[0][1],
                fc1_bias=state_tensors[1][1],
                fc2_weight=state_tensors[2][1],
                fc2_bias=state_tensors[3][1],
            ),
            steps=(
                state_tensors[0][2],
                state_tensors[1][2],
                state_tensors[2][2],
                state_tensors[3][2],
            ),
            models=tuple(models),
            optimizers=tuple(optimizers),
            parameter_ref_groups=tuple(
                tuple(refs) for refs in parameter_ref_groups
            ),  # type: ignore[arg-type]
            state_groups=tuple(tuple(states) for states in state_groups),  # type: ignore[arg-type]
        )

    @property
    def agent_count(self) -> int:
        return self.exp_avg.agent_count

    def state_tensors(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        exp_avg_tensors = self.exp_avg.tensors()
        exp_avg_sq_tensors = self.exp_avg_sq.tensors()
        return (
            (exp_avg_tensors[0], exp_avg_sq_tensors[0], self.steps[0]),
            (exp_avg_tensors[1], exp_avg_sq_tensors[1], self.steps[1]),
            (exp_avg_tensors[2], exp_avg_sq_tensors[2], self.steps[2]),
            (exp_avg_tensors[3], exp_avg_sq_tensors[3], self.steps[3]),
        )

    def references_match_agents(
        self,
        agents: Sequence[object],
        active_ids: Sequence[int],
    ) -> bool:
        if len(agents) != self.agent_count:
            return False
        for agent_id in active_ids:
            try:
                model, optimizer = _model_and_optimizer_from_agent(
                    agents[int(agent_id)],
                    int(agent_id),
                )
            except ValueError:
                return False
            if model is not self.models[int(agent_id)]:
                return False
            if optimizer is not self.optimizers[int(agent_id)]:
                return False
            if not isinstance(optimizer, torch.optim.Adam):
                return False
            if not _adam_optimizer_references_are_supported(
                optimizer,
                self.models[int(agent_id)],
            ):
                return False
        return True

    def active_reference_data(
        self,
        active_ids: Sequence[int],
        *,
        all_agents_active: bool,
    ) -> tuple[
        list[nn.Module],
        list[torch.optim.Adam],
        tuple[list[torch.Tensor], ...],
        tuple[list[dict[str, torch.Tensor]], ...],
    ]:
        if all_agents_active:
            return (
                list(self.models),
                list(self.optimizers),
                tuple(list(refs) for refs in self.parameter_ref_groups),
                tuple(list(states) for states in self.state_groups),
            )
        ids = [int(agent_id) for agent_id in active_ids]
        return (
            [self.models[agent_id] for agent_id in ids],
            [self.optimizers[agent_id] for agent_id in ids],
            tuple(
                [refs[agent_id] for agent_id in ids]
                for refs in self.parameter_ref_groups
            ),
            tuple(
                [states[agent_id] for agent_id in ids]
                for states in self.state_groups
            ),
        )

    def synchronize_agent_state(
        self,
        parameters: "BatchedMLPParameters",
    ) -> None:
        """Copy cached batched parameters and Adam state back to agent objects."""

        if parameters.agent_count != self.agent_count:
            raise ValueError(
                "Parameter count must match Adam state cache count; "
                f"got {parameters.agent_count} for {self.agent_count} agents"
            )
        with torch.no_grad():
            for (
                parameter_values,
                parameter_refs,
                state_cache_tensors,
                optimizer_states,
            ) in zip(
                parameters.tensors(),
                self.parameter_ref_groups,
                self.state_tensors(),
                self.state_groups,
                strict=True,
            ):
                exp_avg, exp_avg_sq, steps = state_cache_tensors
                _copy_tensor_list_(
                    list(parameter_refs),
                    list(parameter_values.detach().unbind(0)),
                )
                _copy_tensor_list_(
                    [state["exp_avg"] for state in optimizer_states],
                    list(exp_avg.detach().unbind(0)),
                )
                _copy_tensor_list_(
                    [state["exp_avg_sq"] for state in optimizer_states],
                    list(exp_avg_sq.detach().unbind(0)),
                )
                _copy_tensor_list_(
                    [state["step"] for state in optimizer_states],
                    list(steps.detach().unbind(0)),
                )


@dataclass
class TensorBatchedMLPRuntime:
    """TensorPolicyRuntime implementation for same-shaped ReLU MLP + Adam agents."""

    parameters: BatchedMLPParameters
    exp_avg: BatchedMLPParameters
    exp_avg_sq: BatchedMLPParameters
    steps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    shared_step_groups: bool = False

    @classmethod
    def from_agents(
        cls,
        agents: Sequence[object],
        *,
        device: torch.device | str | None = None,
    ) -> "TensorBatchedMLPRuntime":
        models: list[nn.Module] = []
        optimizers: list[torch.optim.Adam] = []
        for agent_id, agent in enumerate(agents):
            model, optimizer = _model_and_optimizer_from_agent(agent, agent_id)
            if not isinstance(optimizer, torch.optim.Adam):
                raise ValueError("Tensor batched MLP runtime requires Adam optimizers")
            if not _adam_optimizer_is_supported(optimizer, model):
                raise ValueError("Unsupported Adam optimizer configuration")
            models.append(model)
            optimizers.append(optimizer)
        if not models:
            raise ValueError("At least one agent is required")

        reference_group = optimizers[0].param_groups[0]
        for optimizer in optimizers[1:]:
            if not _adam_groups_match(reference_group, optimizer.param_groups[0]):
                raise ValueError("All tensor batched Adam optimizers must share hyperparameters")

        resolved_device = (
            models[0].fc1.weight.device
            if device is None
            else torch.device(device)
        )
        parameters = BatchedMLPParameters.from_models(models, device=resolved_device)
        parameter_ref_groups = _batched_mlp_parameter_refs(models)
        state_groups: list[list[dict[str, torch.Tensor]]] = []
        for refs in parameter_ref_groups:
            states = []
            for optimizer, parameter in zip(optimizers, refs, strict=True):
                state = _adam_state_for_parameter(optimizer, parameter)
                if state is None:
                    raise ValueError("Unsupported Adam optimizer state")
                states.append(state)
            state_groups.append(states)

        state_tensors = [
            (
                _stack_adam_state_group(
                    states,
                    "exp_avg",
                    device=resolved_device,
                    dtype=refs[0].dtype,
                ),
                _stack_adam_state_group(
                    states,
                    "exp_avg_sq",
                    device=resolved_device,
                    dtype=refs[0].dtype,
                ),
                _stack_adam_state_group(
                    states,
                    "step",
                    device=resolved_device,
                    dtype=refs[0].dtype,
                ),
            )
            for states, refs in zip(
                state_groups,
                parameter_ref_groups,
                strict=True,
            )
        ]
        beta1, beta2 = reference_group["betas"]
        steps = (
            state_tensors[0][2],
            state_tensors[1][2],
            state_tensors[2][2],
            state_tensors[3][2],
        )
        return cls(
            parameters=parameters,
            exp_avg=BatchedMLPParameters(
                fc1_weight=state_tensors[0][0],
                fc1_bias=state_tensors[1][0],
                fc2_weight=state_tensors[2][0],
                fc2_bias=state_tensors[3][0],
            ),
            exp_avg_sq=BatchedMLPParameters(
                fc1_weight=state_tensors[0][1],
                fc1_bias=state_tensors[1][1],
                fc2_weight=state_tensors[2][1],
                fc2_bias=state_tensors[3][1],
            ),
            steps=steps,
            lr=float(reference_group["lr"]),
            betas=(float(beta1), float(beta2)),
            eps=float(reference_group["eps"]),
            weight_decay=float(reference_group["weight_decay"]),
            shared_step_groups=_adam_step_groups_are_equal(steps),
        )

    @property
    def device(self) -> torch.device:
        return self.parameters.device

    @property
    def agent_count(self) -> int:
        return self.parameters.agent_count

    @property
    def input_dim(self) -> int:
        return int(self.parameters.fc1_weight.shape[2])

    @property
    def hidden_dim(self) -> int:
        return int(self.parameters.fc1_weight.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.parameters.fc2_weight.shape[1])

    def state_tensors(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        exp_avg_tensors = self.exp_avg.tensors()
        exp_avg_sq_tensors = self.exp_avg_sq.tensors()
        return (
            (exp_avg_tensors[0], exp_avg_sq_tensors[0], self.steps[0]),
            (exp_avg_tensors[1], exp_avg_sq_tensors[1], self.steps[1]),
            (exp_avg_tensors[2], exp_avg_sq_tensors[2], self.steps[2]),
            (exp_avg_tensors[3], exp_avg_sq_tensors[3], self.steps[3]),
        )

    @torch.no_grad()
    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.parameters.logits(observations)

    @torch.no_grad()
    def probabilities(
        self,
        observations: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self.parameters.probabilities(
            observations,
            temperature=temperature,
        )

    def trainable_parameters(self) -> BatchedMLPParameters:
        return self.parameters.trainable_view()

    def apply_loss_gradients(
        self,
        parameters: BatchedMLPParameters,
        losses: torch.Tensor,
        *,
        active_agent_ids: Sequence[int] | None = None,
        timing_prefix: str | None = None,
        timing_recorder: AcceleratorTimingRecorder | None = None,
        timing_synchronizer: AcceleratorTimingSynchronizer | None = None,
    ) -> BatchedMLPUpdateResult:
        """Backprop per-agent losses and update only runtime-owned tensors."""

        if losses.ndim != 1:
            raise ValueError("Tensor batched MLP losses must have shape [agents]")
        if parameters.agent_count != self.agent_count:
            raise ValueError(
                "Parameter count must match runtime agent count; "
                f"got {parameters.agent_count} for {self.agent_count} agents"
            )
        if parameters.device != self.device:
            raise ValueError("Tensor batched update parameters must use runtime device")
        if losses.shape[0] != self.agent_count:
            raise ValueError(
                "Loss count must match runtime agent count; "
                f"got {losses.shape[0]} losses for {self.agent_count} agents"
            )
        if active_agent_ids is None:
            active_ids: list[int] | None = None
            all_agents_active = True
        else:
            active_ids = [int(agent_id) for agent_id in active_agent_ids]
            for agent_id in active_ids:
                if not 0 <= agent_id < self.agent_count:
                    raise ValueError(f"Active agent id out of range: {agent_id}")
            all_agents_active = len(active_ids) == self.agent_count and all(
                agent_id == index for index, agent_id in enumerate(active_ids)
            )

        if active_ids is not None and not active_ids:
            return BatchedMLPUpdateResult(
                losses=TensorBackedLossVector.zeros(
                    self.agent_count,
                    device=losses.device,
                    dtype=losses.dtype,
                ),
                updated_parameters=self.parameters,
                used_batched_optimizer=True,
            )

        if all_agents_active:
            loss_total = losses.sum()
            active_index = None
        else:
            if active_ids is None:
                raise RuntimeError("Partial tensor update requires active agent ids")
            active_index = torch.as_tensor(
                active_ids,
                dtype=torch.long,
                device=losses.device,
            )
            loss_total = losses.index_select(0, active_index).sum()
        with _timed_accelerator_stage(
            timing_recorder,
            timing_synchronizer,
            _timing_stage(timing_prefix, "autograd_grad"),
        ):
            gradients = torch.autograd.grad(loss_total, parameters.tensors())

        with _timed_accelerator_stage(
            timing_recorder,
            timing_synchronizer,
            _timing_stage(timing_prefix, "adam_update"),
        ):
            with torch.no_grad():
                _apply_tensor_batched_adam_update_(
                    parameters=parameters,
                    gradients=gradients,
                    exp_avg=self.exp_avg,
                    exp_avg_sq=self.exp_avg_sq,
                    steps=self.steps,
                    lr=self.lr,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                    active_index=active_index,
                    shared_step_groups=self.shared_step_groups,
                )
        self.parameters = parameters.detached()
        return BatchedMLPUpdateResult(
            losses=TensorBackedLossVector.from_tensor(
                losses,
                active_agent_ids=None if all_agents_active else active_ids,
                agent_count=self.agent_count,
            ),
            updated_parameters=self.parameters,
            used_batched_optimizer=True,
        )

    def flush_to_agents(self, agents: Sequence[object]) -> None:
        """Copy runtime parameters and Adam state back to compatible agents."""

        if len(agents) != self.agent_count:
            raise ValueError(
                "Agent count must match tensor runtime count; "
                f"got {len(agents)} agents for {self.agent_count} runtime entries"
            )
        models: list[nn.Module] = []
        optimizers: list[torch.optim.Adam] = []
        for agent_id, agent in enumerate(agents):
            model, optimizer = _model_and_optimizer_from_agent(agent, agent_id)
            if not isinstance(optimizer, torch.optim.Adam):
                raise ValueError("Tensor batched MLP runtime requires Adam optimizers")
            if not _adam_optimizer_is_supported(optimizer, model):
                raise ValueError("Unsupported Adam optimizer configuration")
            if not _adam_groups_match(
                {
                    "lr": self.lr,
                    "betas": self.betas,
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                },
                optimizer.param_groups[0],
            ):
                raise ValueError("Agent Adam hyperparameters differ from tensor runtime")
            models.append(model)
            optimizers.append(optimizer)

        parameter_ref_groups = _batched_mlp_parameter_refs(models)
        state_groups: list[list[dict[str, torch.Tensor]]] = []
        for refs in parameter_ref_groups:
            states = []
            for optimizer, parameter in zip(optimizers, refs, strict=True):
                state = _adam_state_for_parameter(optimizer, parameter)
                if state is None:
                    raise ValueError("Unsupported Adam optimizer state")
                states.append(state)
            state_groups.append(states)

        with torch.no_grad():
            for (
                parameter_values,
                parameter_refs,
                state_cache_tensors,
                optimizer_states,
            ) in zip(
                self.parameters.tensors(),
                parameter_ref_groups,
                self.state_tensors(),
                state_groups,
                strict=True,
            ):
                exp_avg, exp_avg_sq, steps = state_cache_tensors
                _copy_tensor_list_(
                    list(parameter_refs),
                    list(parameter_values.detach().unbind(0)),
                )
                _copy_tensor_list_(
                    [state["exp_avg"] for state in optimizer_states],
                    list(exp_avg.detach().unbind(0)),
                )
                _copy_tensor_list_(
                    [state["exp_avg_sq"] for state in optimizer_states],
                    list(exp_avg_sq.detach().unbind(0)),
                )
                _copy_tensor_list_(
                    [state["step"] for state in optimizer_states],
                    list(steps.detach().unbind(0)),
                )


@dataclass
class BatchedMLPPolicyCache:
    """Inference-only cached batched MLP snapshot for per-agent policies."""

    parameters: BatchedMLPParameters

    @classmethod
    def from_models(
        cls,
        models: Sequence[nn.Module],
        *,
        device: torch.device | str | None = None,
    ) -> "BatchedMLPPolicyCache":
        return cls(BatchedMLPParameters.from_models(models, device=device))

    @classmethod
    def from_agents(
        cls,
        agents: Sequence[object],
        *,
        device: torch.device | str | None = None,
    ) -> "BatchedMLPPolicyCache":
        return cls.from_models(_models_from_agents(agents), device=device)

    @property
    def device(self) -> torch.device:
        return self.parameters.device

    @property
    def agent_count(self) -> int:
        return self.parameters.agent_count

    def refresh(self, agents: Sequence[object]) -> None:
        """Refresh the inference snapshot from current agent model weights."""

        self.parameters = BatchedMLPParameters.from_models(
            _models_from_agents(agents),
            device=self.device,
        )

    @torch.no_grad()
    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.parameters.logits(observations)

    @torch.no_grad()
    def probabilities(
        self,
        observations: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self.parameters.probabilities(
            observations,
            temperature=temperature,
        )


def _models_from_agents(agents: Sequence[object]) -> list[nn.Module]:
    models: list[nn.Module] = []
    for index, agent in enumerate(agents):
        model = agent if isinstance(agent, nn.Module) else getattr(agent, "model", None)
        if not isinstance(model, nn.Module):
            raise ValueError(
                "Batched MLP policy cache agents must expose torch Module models; "
                f"agent {index} does not"
            )
        models.append(model)
    return models


def _parameter_snapshot(
    parameter: torch.Tensor,
    device: torch.device,
    *,
    requires_grad: bool,
) -> torch.Tensor:
    snapshot = parameter.detach()
    if requires_grad:
        snapshot = snapshot.clone()
    snapshot = snapshot.to(device)
    if requires_grad:
        snapshot.requires_grad_(True)
    return snapshot


def _validate_batched_mlp_model(model: nn.Module) -> None:
    missing = [
        name
        for name in ("fc1", "activation", "fc2")
        if not hasattr(model, name)
    ]
    if missing:
        raise ValueError(f"Model is missing batched MLP field(s): {', '.join(missing)}")
    if not isinstance(model.fc1, nn.Linear) or not isinstance(model.fc2, nn.Linear):
        raise ValueError("Batched MLP models require Linear fc1 and fc2 layers")
    if not isinstance(model.activation, nn.ReLU):
        raise ValueError("Batched MLP models currently require ReLU activation")
    if model.fc2.in_features != model.fc1.out_features:
        raise ValueError("Model fc2 input dimension must match fc1 output dimension")


@torch.no_grad()
def batched_mlp_policy_probs(
    models: Sequence[nn.Module],
    observations: torch.Tensor,
    *,
    temperature: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Evaluate one same-shaped MLP policy per agent in a single tensor kernel."""

    cache = BatchedMLPPolicyCache.from_models(models, device=device)
    return cache.probabilities(observations, temperature=temperature)


def trainable_batched_mlp_parameters(
    agents: Sequence[object],
    *,
    device: torch.device | str | None = None,
) -> BatchedMLPParameters:
    """Snapshot per-agent MLP parameters for a batched autograd pass."""

    return BatchedMLPParameters.from_models(
        _models_from_agents(agents),
        device=device,
        requires_grad=True,
    )


def batched_binary_policy_gradient_losses(
    parameters: BatchedMLPParameters,
    observations: torch.Tensor,
    *,
    actions: Sequence[int] | torch.Tensor,
    advantages: Sequence[float] | torch.Tensor,
    entropy_beta: float,
) -> torch.Tensor:
    """Return per-agent binary policy-gradient losses for batched MLP policies."""

    logits = parameters.logits(observations)
    action_tensor = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=logits.device,
    )
    advantage_tensor = torch.as_tensor(
        advantages,
        dtype=logits.dtype,
        device=logits.device,
    )
    if logits.shape[1] == 2:
        logit_delta = logits[:, 1] - logits[:, 0]
        log_prob_one = F.logsigmoid(logit_delta)
        log_prob_zero = F.logsigmoid(-logit_delta)
        prob_one = torch.sigmoid(logit_delta)
        entropy = -(
            (1.0 - prob_one) * log_prob_zero
            + prob_one * log_prob_one
        )
        action_log_probs = torch.where(
            action_tensor == 1,
            log_prob_one,
            log_prob_zero,
        )
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        action_log_probs = log_probs.gather(1, action_tensor[:, None]).squeeze(1)
    return -advantage_tensor * action_log_probs - entropy_beta * entropy


def batched_distribution_cross_entropy_losses(
    parameters: BatchedMLPParameters,
    observations: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_mode: str = "cross_entropy",
) -> torch.Tensor:
    """Return per-agent distribution distillation losses."""

    logits = parameters.logits(observations)
    target_tensor = targets.detach().to(
        device=logits.device,
        dtype=logits.dtype,
    )
    log_probs = F.log_softmax(logits, dim=-1)
    if loss_mode == "cross_entropy":
        return -(target_tensor * log_probs).sum(dim=-1)
    if loss_mode == "kl":
        cross_entropy = -(target_tensor * log_probs).sum(dim=-1)
        target_entropy = -(target_tensor * target_tensor.clamp_min(1e-12).log()).sum(
            dim=-1,
        )
        return cross_entropy - target_entropy.detach()
    raise ValueError(f"Unsupported distribution loss mode: {loss_mode}")


def apply_batched_mlp_loss_gradients(
    agents: Sequence[object],
    parameters: BatchedMLPParameters,
    losses: torch.Tensor,
    *,
    active_agent_ids: Sequence[int] | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
    timing_prefix: str | None = None,
    timing_recorder: AcceleratorTimingRecorder | None = None,
    timing_synchronizer: AcceleratorTimingSynchronizer | None = None,
) -> list[float]:
    return list(
        apply_batched_mlp_loss_gradients_with_result(
            agents=agents,
            parameters=parameters,
            losses=losses,
            active_agent_ids=active_agent_ids,
            adam_state_cache=adam_state_cache,
            synchronize_model_parameters=synchronize_model_parameters,
            synchronize_optimizer_states=synchronize_optimizer_states,
            timing_prefix=timing_prefix,
            timing_recorder=timing_recorder,
            timing_synchronizer=timing_synchronizer,
        ).losses
    )


def apply_batched_mlp_loss_gradients_with_result(
    agents: Sequence[object],
    parameters: BatchedMLPParameters,
    losses: torch.Tensor,
    *,
    active_agent_ids: Sequence[int] | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
    timing_prefix: str | None = None,
    timing_recorder: AcceleratorTimingRecorder | None = None,
    timing_synchronizer: AcceleratorTimingSynchronizer | None = None,
) -> BatchedMLPUpdateResult:
    """Backprop batched per-agent losses into existing agent optimizers.

    The batched parameter snapshot owns the autograd graph. This helper uses an
    equivalent batched Adam update when possible, and otherwise copies each
    active agent's gradient slice back before calling the original optimizer in
    agent-id order.
    """

    if losses.ndim != 1:
        raise ValueError("Batched MLP losses must have shape [agents]")
    if losses.shape[0] != parameters.agent_count:
        raise ValueError(
            "Loss count must match model count; "
            f"got {losses.shape[0]} losses for {parameters.agent_count} models"
        )
    if len(agents) != parameters.agent_count:
        raise ValueError(
            "Agent count must match parameter count; "
            f"got {len(agents)} agents for {parameters.agent_count} models"
        )

    full_active_ids = list(range(parameters.agent_count))
    if active_agent_ids is None:
        active_ids = full_active_ids
    else:
        active_ids = [int(agent_id) for agent_id in active_agent_ids]
    for agent_id in active_ids:
        if not 0 <= agent_id < parameters.agent_count:
            raise ValueError(f"Active agent id out of range: {agent_id}")

    if not active_ids:
        return BatchedMLPUpdateResult(
            losses=TensorBackedLossVector.zeros(
                parameters.agent_count,
                device=losses.device,
                dtype=losses.dtype,
            ),
            updated_parameters=parameters.detached(),
            used_batched_optimizer=True,
        )

    all_agents_active = active_ids == full_active_ids
    if all_agents_active:
        loss_total = losses.sum()
    else:
        active_index = torch.as_tensor(
            active_ids,
            dtype=torch.long,
            device=losses.device,
        )
        loss_total = losses.index_select(0, active_index).sum()
    with _timed_accelerator_stage(
        timing_recorder,
        timing_synchronizer,
        _timing_stage(timing_prefix, "autograd_grad"),
    ):
        gradients = torch.autograd.grad(loss_total, parameters.tensors())

    with _timed_accelerator_stage(
        timing_recorder,
        timing_synchronizer,
        _timing_stage(timing_prefix, "adam_update"),
    ):
        used_batched_optimizer = _apply_batched_adam_update(
            agents=agents,
            parameters=parameters,
            gradients=gradients,
            active_ids=active_ids,
            all_agents_active=all_agents_active,
            adam_state_cache=adam_state_cache,
            synchronize_model_parameters=synchronize_model_parameters,
            synchronize_optimizer_states=synchronize_optimizer_states,
        )
    if used_batched_optimizer:
        return BatchedMLPUpdateResult(
            losses=TensorBackedLossVector.from_tensor(
                losses,
                active_agent_ids=None if all_agents_active else active_ids,
                agent_count=parameters.agent_count,
            ),
            updated_parameters=parameters.detached(),
            used_batched_optimizer=True,
        )

    for agent_id in active_ids:
        model, optimizer = _model_and_optimizer_from_agent(agents[agent_id], agent_id)
        optimizer.zero_grad()
        _assign_parameter_grad(model.fc1.weight, gradients[0][agent_id])
        _assign_parameter_grad(model.fc1.bias, gradients[1][agent_id])
        _assign_parameter_grad(model.fc2.weight, gradients[2][agent_id])
        _assign_parameter_grad(model.fc2.bias, gradients[3][agent_id])
        optimizer.step()

    return BatchedMLPUpdateResult(
        losses=TensorBackedLossVector.from_tensor(
            losses,
            active_agent_ids=None if all_agents_active else active_ids,
            agent_count=parameters.agent_count,
        ),
        updated_parameters=None,
        used_batched_optimizer=False,
    )


def _timing_stage(prefix: str | None, suffix: str) -> str | None:
    if not prefix:
        return None
    return f"{prefix}_{suffix}"


@dataclass
class _AcceleratorTimer:
    recorder: AcceleratorTimingRecorder | None
    synchronizer: AcceleratorTimingSynchronizer | None
    stage: str | None
    start: float = 0.0

    def __enter__(self) -> None:
        if self.recorder is None or self.stage is None:
            return
        if self.synchronizer is not None:
            self.synchronizer()
        self.start = time.perf_counter()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self.recorder is None or self.stage is None:
            return
        if self.synchronizer is not None:
            self.synchronizer()
        self.recorder(self.stage, time.perf_counter() - self.start)


def _timed_accelerator_stage(
    recorder: AcceleratorTimingRecorder | None,
    synchronizer: AcceleratorTimingSynchronizer | None,
    stage: str | None,
) -> _AcceleratorTimer:
    return _AcceleratorTimer(
        recorder=recorder,
        synchronizer=synchronizer,
        stage=stage,
    )


def _model_and_optimizer_from_agent(
    agent: object,
    agent_id: int,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = agent if isinstance(agent, nn.Module) else getattr(agent, "model", None)
    optimizer = getattr(agent, "optimizer", None)
    if not isinstance(model, nn.Module) or optimizer is None:
        raise ValueError(
            "Batched gradient application requires agents with model and optimizer; "
            f"agent {agent_id} does not"
        )
    return model, optimizer


def _apply_batched_adam_update(
    *,
    agents: Sequence[object],
    parameters: BatchedMLPParameters,
    gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    active_ids: list[int],
    all_agents_active: bool,
    adam_state_cache: BatchedAdamStateCache | None,
    synchronize_model_parameters: bool,
    synchronize_optimizer_states: bool,
) -> bool:
    if not active_ids:
        return True
    if adam_state_cache is not None and not _adam_state_cache_matches_parameters(
        adam_state_cache,
        parameters,
    ):
        return False
    if adam_state_cache is not None:
        if not adam_state_cache.references_match_agents(agents, active_ids):
            return False
        models, optimizers, parameter_ref_groups, state_groups = (
            adam_state_cache.active_reference_data(
                active_ids,
                all_agents_active=all_agents_active,
            )
        )
    else:
        models = []
        optimizers = []
        for agent_id in active_ids:
            model, optimizer = _model_and_optimizer_from_agent(agents[agent_id], agent_id)
            if not isinstance(optimizer, torch.optim.Adam):
                return False
            if not _adam_optimizer_is_supported(optimizer, model):
                return False
            models.append(model)
            optimizers.append(optimizer)

        parameter_ref_groups = _batched_mlp_parameter_refs(models)
        state_groups = []
        for refs in parameter_ref_groups:
            states = []
            for optimizer, parameter in zip(optimizers, refs, strict=True):
                state = _adam_state_for_parameter(optimizer, parameter)
                if state is None:
                    return False
                states.append(state)
            state_groups.append(states)
        state_groups = tuple(state_groups)

    reference_group = optimizers[0].param_groups[0]
    for optimizer in optimizers[1:]:
        if not _adam_groups_match(reference_group, optimizer.param_groups[0]):
            return False

    with torch.no_grad():
        active_index = None
        if not all_agents_active:
            active_index = torch.as_tensor(
                active_ids,
                dtype=torch.long,
                device=parameters.device,
            )
        if (
            adam_state_cache is not None
            and _supports_batched_adam_flat_update(parameters.device)
            and _apply_batched_adam_flat_update(
                parameters=parameters,
                gradients=gradients,
                parameter_ref_groups=parameter_ref_groups,
                state_groups=state_groups,
                state_cache_tensor_groups=adam_state_cache.state_tensors(),
                group=reference_group,
                active_index=active_index,
                synchronize_model_parameters=synchronize_model_parameters,
                synchronize_optimizer_states=synchronize_optimizer_states,
            )
        ):
            for model, optimizer in zip(models, optimizers, strict=True):
                if any(
                    parameter.grad is not None
                    for parameter in _model_parameter_refs(model)
                ):
                    optimizer.zero_grad(set_to_none=True)
            return True
        for (
            state_cache_tensors,
            parameter_values,
            parameter_refs,
            gradient_values,
            optimizer_states,
        ) in zip(
            _empty_adam_state_cache_tensors()
            if adam_state_cache is None
            else adam_state_cache.state_tensors(),
            parameters.tensors(),
            parameter_ref_groups,
            gradients,
            state_groups,
            strict=True,
        ):
            if not _apply_batched_adam_parameter_update(
                parameter_values=parameter_values,
                parameter_refs=parameter_refs,
                gradient_values=gradient_values,
                optimizers=optimizers,
                group=reference_group,
                active_index=active_index,
                state_cache_tensors=state_cache_tensors,
                optimizer_states=optimizer_states,
                synchronize_model_parameters=synchronize_model_parameters,
                synchronize_optimizer_states=synchronize_optimizer_states,
            ):
                return False
        for model, optimizer in zip(models, optimizers, strict=True):
            if any(parameter.grad is not None for parameter in _model_parameter_refs(model)):
                optimizer.zero_grad(set_to_none=True)
    return True


def _apply_batched_adam_flat_update(
    *,
    parameters: BatchedMLPParameters,
    gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    parameter_ref_groups: tuple[list[torch.Tensor], ...],
    state_groups: tuple[list[dict[str, torch.Tensor]], ...],
    state_cache_tensor_groups: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    group: dict[str, object],
    active_index: torch.Tensor | None,
    synchronize_model_parameters: bool,
    synchronize_optimizer_states: bool,
) -> bool:
    parameter_tensors = parameters.tensors()
    update_device = parameter_tensors[0].device
    update_dtype = parameter_tensors[0].dtype
    if any(
        tensor.device != update_device or tensor.dtype != update_dtype
        for tensor in parameter_tensors
    ):
        return False
    if any(
        gradient.device != update_device
        for gradient in gradients
    ):
        return False

    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    beta1 = float(beta1)
    beta2 = float(beta2)
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])

    active_count = (
        parameters.agent_count
        if active_index is None
        else int(active_index.numel())
    )
    if any(len(refs) != active_count for refs in parameter_ref_groups):
        return False
    if any(len(states) != active_count for states in state_groups):
        return False
    flat_values: list[torch.Tensor] = []
    flat_grads: list[torch.Tensor] = []
    flat_exp_avg: list[torch.Tensor] = []
    flat_exp_avg_sq: list[torch.Tensor] = []
    step_groups: list[torch.Tensor] = []
    flat_steps: list[torch.Tensor] = []
    widths: list[int] = []

    for (
        parameter_values,
        gradient_values,
        state_cache_tensors,
    ) in zip(
        parameter_tensors,
        gradients,
        state_cache_tensor_groups,
        strict=True,
    ):
        cached_exp_avg, cached_exp_avg_sq, cached_steps = state_cache_tensors
        if (
            cached_exp_avg.device != update_device
            or cached_exp_avg.dtype != update_dtype
            or cached_exp_avg_sq.device != update_device
            or cached_exp_avg_sq.dtype != update_dtype
            or cached_steps.device != update_device
            or cached_steps.dtype != update_dtype
        ):
            return False
        values = parameter_values.detach()
        grads = gradient_values.detach()
        exp_avg = cached_exp_avg
        exp_avg_sq = cached_exp_avg_sq
        steps = cached_steps
        if active_index is not None:
            values = values.index_select(0, active_index)
            grads = grads.index_select(0, active_index)
            exp_avg = exp_avg.index_select(0, active_index)
            exp_avg_sq = exp_avg_sq.index_select(0, active_index)
            steps = steps.index_select(0, active_index)
        grads = grads.to(device=update_device, dtype=update_dtype)
        width = int(values[0].numel()) if active_count > 0 else int(values[0:].numel())
        widths.append(width)
        flat_values.append(values.reshape(active_count, width))
        flat_grads.append(grads.reshape(active_count, width))
        flat_exp_avg.append(exp_avg.reshape(active_count, width))
        flat_exp_avg_sq.append(exp_avg_sq.reshape(active_count, width))
        next_steps = steps + 1.0
        step_groups.append(next_steps)
        flat_steps.append(next_steps[:, None].expand(active_count, width))

    values = torch.cat(flat_values, dim=1)
    grads = torch.cat(flat_grads, dim=1)
    exp_avg = torch.cat(flat_exp_avg, dim=1)
    exp_avg_sq = torch.cat(flat_exp_avg_sq, dim=1)
    steps = torch.cat(flat_steps, dim=1)
    if weight_decay:
        grads = grads.add(values, alpha=weight_decay)

    exp_avg.mul_(beta1).add_(grads, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1.0 - beta2)
    bias_correction1 = 1.0 - torch.pow(beta1, steps)
    bias_correction2 = 1.0 - torch.pow(beta2, steps)
    step_size = lr / bias_correction1
    denominator = exp_avg_sq.sqrt() / bias_correction2.sqrt()
    denominator = denominator.add(eps)
    updated_values = values - step_size * exp_avg / denominator

    value_chunks = updated_values.split(widths, dim=1)
    exp_avg_chunks = exp_avg.split(widths, dim=1)
    exp_avg_sq_chunks = exp_avg_sq.split(widths, dim=1)
    for (
        parameter_values,
        parameter_refs,
        optimizer_states,
        state_cache_tensors,
        value_chunk,
        exp_avg_chunk,
        exp_avg_sq_chunk,
        step_group,
    ) in zip(
        parameter_tensors,
        parameter_ref_groups,
        state_groups,
        state_cache_tensor_groups,
        value_chunks,
        exp_avg_chunks,
        exp_avg_sq_chunks,
        step_groups,
        strict=True,
    ):
        active_shape = (active_count, *parameter_values.shape[1:])
        parameter_update = value_chunk.reshape(active_shape)
        exp_avg_update = exp_avg_chunk.reshape(active_shape)
        exp_avg_sq_update = exp_avg_sq_chunk.reshape(active_shape)

        _update_adam_state_cache_tensors_(
            state_cache_tensors,
            active_index=active_index,
            exp_avg=exp_avg_update,
            exp_avg_sq=exp_avg_sq_update,
            steps=step_group,
        )
        if synchronize_model_parameters:
            _copy_tensor_list_(parameter_refs, list(parameter_update.unbind(0)))
        if synchronize_optimizer_states:
            _copy_tensor_list_(
                [state["exp_avg"] for state in optimizer_states],
                list(exp_avg_update.unbind(0)),
            )
            _copy_tensor_list_(
                [state["exp_avg_sq"] for state in optimizer_states],
                list(exp_avg_sq_update.unbind(0)),
            )
            _copy_tensor_list_(
                [state["step"] for state in optimizer_states],
                list(step_group.unbind(0)),
            )
        if active_index is None:
            parameter_values.copy_(parameter_update)
        else:
            parameter_values.index_copy_(0, active_index, parameter_update)
    return True


def _adam_optimizer_is_supported(
    optimizer: torch.optim.Adam,
    model: nn.Module,
) -> bool:
    if len(optimizer.param_groups) != 1:
        return False
    group = optimizer.param_groups[0]
    group_params = list(group["params"])
    model_params = list(_model_parameter_refs(model))
    if len(group_params) != len(model_params):
        return False
    if not all(
        actual is expected
        for actual, expected in zip(
            group_params,
            model_params,
            strict=True,
        )
    ):
        return False
    return _adam_optimizer_group_is_supported(group)


def _adam_optimizer_references_are_supported(
    optimizer: torch.optim.Adam,
    model: nn.Module,
) -> bool:
    return _adam_optimizer_is_supported(optimizer, model)


def _adam_optimizer_group_is_supported(group: dict[str, object]) -> bool:
    if bool(group.get("amsgrad", False)):
        return False
    if bool(group.get("maximize", False)):
        return False
    if bool(group.get("capturable", False)):
        return False
    if bool(group.get("differentiable", False)):
        return False
    if bool(group.get("fused", False)):
        return False
    if bool(group.get("decoupled_weight_decay", False)):
        return False
    foreach = group.get("foreach", None)
    if foreach not in {None, False}:
        return False
    return True


def _adam_groups_match(
    first: dict[str, object],
    second: dict[str, object],
) -> bool:
    keys = ("lr", "betas", "eps", "weight_decay")
    return all(first.get(key) == second.get(key) for key in keys)


def _apply_batched_adam_parameter_update(
    *,
    parameter_values: torch.Tensor,
    parameter_refs: list[torch.Tensor],
    gradient_values: torch.Tensor,
    optimizers: list[torch.optim.Adam],
    group: dict[str, object],
    active_index: torch.Tensor | None,
    state_cache_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    optimizer_states: list[dict[str, torch.Tensor]] | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
) -> bool:
    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    beta1 = float(beta1)
    beta2 = float(beta2)
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])

    if optimizer_states is None:
        states = []
        for optimizer, parameter in zip(optimizers, parameter_refs, strict=True):
            state = _adam_state_for_parameter(optimizer, parameter)
            if state is None:
                return False
            states.append(state)
    else:
        states = optimizer_states

    update_device = parameter_values.device
    update_dtype = parameter_values.dtype
    if state_cache_tensors is None:
        exp_avg = torch.stack(
            [
                state["exp_avg"].detach().to(device=update_device, dtype=update_dtype)
                for state in states
            ],
            dim=0,
        )
        exp_avg_sq = torch.stack(
            [
                state["exp_avg_sq"].detach().to(
                    device=update_device,
                    dtype=update_dtype,
                )
                for state in states
            ],
            dim=0,
        )
        steps = torch.stack(
            [
                state["step"].detach().to(device=update_device, dtype=update_dtype)
                for state in states
            ],
            dim=0,
        )
    else:
        cached_exp_avg, cached_exp_avg_sq, cached_steps = state_cache_tensors
        exp_avg = cached_exp_avg
        exp_avg_sq = cached_exp_avg_sq
        steps = cached_steps
        if active_index is not None:
            exp_avg = exp_avg.index_select(0, active_index)
            exp_avg_sq = exp_avg_sq.index_select(0, active_index)
            steps = steps.index_select(0, active_index)

    values = parameter_values.detach()
    grads = gradient_values.detach().to(
        device=update_device,
        dtype=update_dtype,
    )
    if active_index is not None:
        values = values.index_select(0, active_index)
        grads = grads.index_select(0, active_index)
    if weight_decay:
        grads = grads.add(values, alpha=weight_decay)

    steps = steps + 1.0
    exp_avg.mul_(beta1).add_(grads, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1.0 - beta2)
    if state_cache_tensors is not None:
        _update_adam_state_cache_tensors_(
            state_cache_tensors,
            active_index=active_index,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            steps=steps,
        )

    correction_shape = (len(parameter_refs),) + (1,) * (grads.ndim - 1)
    bias_correction1 = (1.0 - torch.pow(beta1, steps)).reshape(correction_shape)
    bias_correction2 = (1.0 - torch.pow(beta2, steps)).reshape(correction_shape)
    step_size = lr / bias_correction1
    denominator = exp_avg_sq.sqrt() / bias_correction2.sqrt()
    denominator = denominator.add(eps)
    updated_values = values - step_size * exp_avg / denominator

    if synchronize_model_parameters or state_cache_tensors is None:
        _copy_tensor_list_(parameter_refs, list(updated_values.unbind(0)))
    if synchronize_optimizer_states or state_cache_tensors is None:
        _copy_tensor_list_(
            [state["exp_avg"] for state in states],
            list(exp_avg.unbind(0)),
        )
        _copy_tensor_list_(
            [state["exp_avg_sq"] for state in states],
            list(exp_avg_sq.unbind(0)),
        )
        _copy_tensor_list_(
            [state["step"] for state in states],
            list(steps.unbind(0)),
        )
    updated_parameter_values = updated_values.to(
        device=parameter_values.device,
        dtype=parameter_values.dtype,
    )
    if active_index is None:
        parameter_values.copy_(updated_parameter_values)
    else:
        parameter_values.index_copy_(
            0,
            active_index,
            updated_parameter_values,
        )
    return True


def _apply_tensor_batched_adam_update_(
    *,
    parameters: BatchedMLPParameters,
    gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    exp_avg: BatchedMLPParameters,
    exp_avg_sq: BatchedMLPParameters,
    steps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    active_index: torch.Tensor | None,
    shared_step_groups: bool = False,
) -> None:
    beta1, beta2 = betas
    parameter_tensors = parameters.tensors()
    active_count = (
        parameters.agent_count
        if active_index is None
        else int(active_index.numel())
    )
    shared_bias_correction1: torch.Tensor | None = None
    shared_bias_correction2: torch.Tensor | None = None
    if active_index is None and shared_step_groups:
        shared_steps = steps[0]
        shared_steps.add_(1.0)
        for step_values in steps[1:]:
            step_values.copy_(shared_steps)
        shared_bias_correction1 = 1.0 - torch.pow(beta1, shared_steps)
        shared_bias_correction2 = 1.0 - torch.pow(beta2, shared_steps)

    for (
        parameter_values,
        gradient_values,
        exp_avg_values,
        exp_avg_sq_values,
        step_values,
    ) in zip(
        parameter_tensors,
        gradients,
        exp_avg.tensors(),
        exp_avg_sq.tensors(),
        steps,
        strict=True,
    ):
        update_device = parameter_values.device
        update_dtype = parameter_values.dtype
        grads = gradient_values.detach().to(
            device=update_device,
            dtype=update_dtype,
        )
        if active_index is None:
            values = parameter_values.detach()
            if weight_decay:
                grads = grads.add(values, alpha=weight_decay)

            if shared_bias_correction1 is None:
                step_values.add_(1.0)
            exp_avg_values.mul_(beta1).add_(grads, alpha=1.0 - beta1)
            exp_avg_sq_values.mul_(beta2).addcmul_(grads, grads, value=1.0 - beta2)

            correction_shape = (active_count,) + (1,) * (grads.ndim - 1)
            correction1 = (
                shared_bias_correction1
                if shared_bias_correction1 is not None
                else 1.0 - torch.pow(beta1, step_values)
            )
            correction2 = (
                shared_bias_correction2
                if shared_bias_correction2 is not None
                else 1.0 - torch.pow(beta2, step_values)
            )
            bias_correction1 = correction1.reshape(
                correction_shape,
            )
            bias_correction2 = correction2.reshape(
                correction_shape,
            )
            step_size = lr / bias_correction1
            denominator = exp_avg_sq_values.sqrt() / bias_correction2.sqrt()
            denominator = denominator.add(eps)
            denominator.div_(step_size)
            parameter_values.addcdiv_(exp_avg_values, denominator, value=-1.0)
            continue

        values = parameter_values.detach().index_select(0, active_index)
        grads = grads.index_select(0, active_index)
        active_exp_avg = exp_avg_values.index_select(0, active_index)
        active_exp_avg_sq = exp_avg_sq_values.index_select(0, active_index)
        active_steps = step_values.index_select(0, active_index)
        if weight_decay:
            grads = grads.add(values, alpha=weight_decay)

        active_steps = active_steps + 1.0
        active_exp_avg.mul_(beta1).add_(grads, alpha=1.0 - beta1)
        active_exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1.0 - beta2)

        correction_shape = (active_count,) + (1,) * (grads.ndim - 1)
        bias_correction1 = (1.0 - torch.pow(beta1, active_steps)).reshape(
            correction_shape,
        )
        bias_correction2 = (1.0 - torch.pow(beta2, active_steps)).reshape(
            correction_shape,
        )
        step_size = lr / bias_correction1
        denominator = active_exp_avg_sq.sqrt() / bias_correction2.sqrt()
        denominator = denominator.add(eps)
        updated_values = values - step_size * active_exp_avg / denominator

        parameter_values.index_copy_(0, active_index, updated_values)
        exp_avg_values.index_copy_(0, active_index, active_exp_avg)
        exp_avg_sq_values.index_copy_(0, active_index, active_exp_avg_sq)
        step_values.index_copy_(0, active_index, active_steps)


def _copy_tensor_list_(
    destinations: list[torch.Tensor],
    sources: list[torch.Tensor],
) -> None:
    converted_sources = [
        source.to(device=destination.device, dtype=destination.dtype)
        for destination, source in zip(destinations, sources, strict=True)
    ]
    foreach_copy = getattr(torch, "_foreach_copy_", None)
    if foreach_copy is not None:
        try:
            foreach_copy(destinations, converted_sources)
            return
        except RuntimeError:
            pass
    for destination, source in zip(destinations, converted_sources, strict=True):
        destination.copy_(source)


def _stack_adam_state_group(
    states: list[dict[str, torch.Tensor]],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            state[key].detach().to(device=device, dtype=dtype)
            for state in states
        ],
        dim=0,
    )


def _adam_step_groups_are_equal(
    steps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    reference = steps[0]
    return all(torch.equal(reference, step_values) for step_values in steps[1:])


def _empty_adam_state_cache_tensors() -> tuple[
    None,
    None,
    None,
    None,
]:
    return (None, None, None, None)


def _adam_state_cache_matches_parameters(
    state_cache: BatchedAdamStateCache,
    parameters: BatchedMLPParameters,
) -> bool:
    if state_cache.agent_count != parameters.agent_count:
        return False
    for state_tensor, parameter_tensor in zip(
        state_cache.exp_avg.tensors(),
        parameters.tensors(),
        strict=True,
    ):
        if state_tensor.shape != parameter_tensor.shape:
            return False
        if state_tensor.device != parameter_tensor.device:
            return False
        if state_tensor.dtype != parameter_tensor.dtype:
            return False
    for state_tensor, parameter_tensor in zip(
        state_cache.exp_avg_sq.tensors(),
        parameters.tensors(),
        strict=True,
    ):
        if state_tensor.shape != parameter_tensor.shape:
            return False
        if state_tensor.device != parameter_tensor.device:
            return False
        if state_tensor.dtype != parameter_tensor.dtype:
            return False
    for step_tensor, parameter_tensor in zip(
        state_cache.steps,
        parameters.tensors(),
        strict=True,
    ):
        if step_tensor.shape != parameter_tensor.shape[:1]:
            return False
        if step_tensor.device != parameter_tensor.device:
            return False
        if step_tensor.dtype != parameter_tensor.dtype:
            return False
    return True


def _update_adam_state_cache_tensors_(
    state_cache_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    active_index: torch.Tensor | None,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    steps: torch.Tensor,
) -> None:
    cached_exp_avg, cached_exp_avg_sq, cached_steps = state_cache_tensors
    if active_index is None:
        if cached_exp_avg is not exp_avg:
            cached_exp_avg.copy_(exp_avg)
        if cached_exp_avg_sq is not exp_avg_sq:
            cached_exp_avg_sq.copy_(exp_avg_sq)
        if cached_steps is not steps:
            cached_steps.copy_(steps)
        return
    cached_exp_avg.index_copy_(0, active_index, exp_avg)
    cached_exp_avg_sq.index_copy_(0, active_index, exp_avg_sq)
    cached_steps.index_copy_(0, active_index, steps)


def _adam_state_for_parameter(
    optimizer: torch.optim.Adam,
    parameter: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    state = optimizer.state[parameter]
    if not state:
        state["step"] = torch.zeros((), dtype=torch.float32)
        state["exp_avg"] = torch.zeros_like(
            parameter,
            memory_format=torch.preserve_format,
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter,
            memory_format=torch.preserve_format,
        )
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        return None
    if not all(isinstance(state[key], torch.Tensor) for key in state):
        return None
    return state  # type: ignore[return-value]


def _model_parameter_refs(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return (
        model.fc1.weight,
        model.fc1.bias,
        model.fc2.weight,
        model.fc2.bias,
    )


def _batched_mlp_parameter_refs(
    models: Sequence[nn.Module],
) -> tuple[list[torch.Tensor], ...]:
    return (
        [model.fc1.weight for model in models],
        [model.fc1.bias for model in models],
        [model.fc2.weight for model in models],
        [model.fc2.bias for model in models],
    )


def _assign_parameter_grad(parameter: torch.Tensor, gradient: torch.Tensor) -> None:
    parameter.grad = gradient.detach().to(
        device=parameter.device,
        dtype=parameter.dtype,
    ).clone()
