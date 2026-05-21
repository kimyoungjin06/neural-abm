"""Loss-vector helpers shared by batched training and binary runners."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(eq=False)
class TensorBackedLossVector(Sequence[float]):
    """A per-agent loss vector that stays on-device until values are needed."""

    _values: torch.Tensor
    _cpu_values: list[float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self._values.ndim != 1:
            raise ValueError("TensorBackedLossVector values must be 1D")
        self._values = self._values.detach()

    @classmethod
    def from_tensor(
        cls,
        losses: torch.Tensor,
        *,
        active_agent_ids: Sequence[int] | None = None,
        agent_count: int | None = None,
    ) -> TensorBackedLossVector:
        if losses.ndim != 1:
            raise ValueError("Loss tensor must be 1D")
        detached = losses.detach()
        resolved_count = int(detached.shape[0]) if agent_count is None else agent_count
        if resolved_count < 0:
            raise ValueError("agent_count must be non-negative")

        if active_agent_ids is None:
            if int(detached.shape[0]) != resolved_count:
                raise ValueError("Loss tensor length must match agent_count")
            return cls(detached)

        active_ids = [int(agent_id) for agent_id in active_agent_ids]
        for agent_id in active_ids:
            if not 0 <= agent_id < resolved_count:
                raise ValueError(f"Active agent id out of range: {agent_id}")
        values = torch.zeros(
            resolved_count,
            dtype=detached.dtype,
            device=detached.device,
        )
        if active_ids:
            active_index = torch.as_tensor(
                active_ids,
                dtype=torch.long,
                device=detached.device,
            )
            values.index_copy_(0, active_index, detached.index_select(0, active_index))
        return cls(values)

    @classmethod
    def zeros(
        cls,
        agent_count: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TensorBackedLossVector:
        if agent_count < 0:
            raise ValueError("agent_count must be non-negative")
        return cls(torch.zeros(agent_count, dtype=dtype, device=device))

    @property
    def tensor(self) -> torch.Tensor:
        return self._values

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def __getitem__(self, index: int | slice) -> float | list[float]:
        if isinstance(index, slice):
            return self.tolist()[index]
        return float(self._values[index].detach().cpu())

    def __iter__(self) -> Iterable[float]:
        return iter(self.tolist())

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        array = self.numpy()
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

    def __repr__(self) -> str:
        return repr(self.tolist())

    def tolist(self) -> list[float]:
        if self._cpu_values is None:
            self._cpu_values = [
                float(value) for value in self._values.detach().cpu().tolist()
            ]
        return list(self._cpu_values)

    def numpy(self) -> np.ndarray:
        return np.asarray(self.tolist(), dtype=np.float64)

    def mean(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(self._values.mean().detach().cpu())

    def isfinite_all(self) -> bool:
        return bool(torch.all(torch.isfinite(self._values)).detach().cpu())

    def values_at(self, indices: Iterable[int]) -> list[float]:
        agent_ids = [int(index) for index in indices]
        if not agent_ids:
            return []
        index_tensor = torch.as_tensor(
            agent_ids,
            dtype=torch.long,
            device=self._values.device,
        )
        return [
            float(value)
            for value in self._values.index_select(0, index_tensor)
            .detach()
            .cpu()
            .tolist()
        ]


LossVector = Sequence[float]


def mean_loss_value(losses: Sequence[float] | None) -> float:
    if losses is None or len(losses) == 0:
        return 0.0
    if isinstance(losses, TensorBackedLossVector):
        return losses.mean()
    return float(np.mean(losses))


def loss_values_at(losses: Sequence[float], indices: Iterable[int]) -> list[float]:
    if isinstance(losses, TensorBackedLossVector):
        return losses.values_at(indices)
    return [losses[int(index)] for index in indices]
