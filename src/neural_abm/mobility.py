"""Reusable fixed-cell mobility primitives."""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MobilityParams:
    enabled: bool = False
    rate: float = 0.0
    candidate_pool_size: int = 8
    selection_rule: str = "local_quality"
    move_cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("mobility rate must lie in [0, 1]")
        if self.candidate_pool_size <= 0:
            raise ValueError("mobility candidate_pool_size must be positive")
        if self.selection_rule != "local_quality":
            raise ValueError(
                f"unsupported mobility selection rule: {self.selection_rule}"
            )
        if self.move_cost < 0.0:
            raise ValueError("mobility move_cost must be non-negative")


@dataclass
class MobilityStepResult:
    moved: np.ndarray
    targets: np.ndarray
    gains: np.ndarray

    @classmethod
    def none(cls, agent_count: int) -> "MobilityStepResult":
        return cls(
            moved=np.zeros(agent_count, dtype=bool),
            targets=np.full(agent_count, -1, dtype=np.int64),
            gains=np.zeros(agent_count, dtype=np.float64),
        )


def local_quality(
    quality_signal: np.ndarray,
    neighbors: list[list[int]],
) -> np.ndarray:
    """Average each cell's quality signal over itself and local neighbors."""

    quality = np.zeros(len(quality_signal), dtype=np.float64)
    for cell_id, peer_ids in enumerate(neighbors):
        if peer_ids:
            local_values = np.concatenate(
                [
                    np.asarray([quality_signal[cell_id]], dtype=np.float64),
                    quality_signal[peer_ids],
                ]
            )
            quality[cell_id] = float(np.mean(local_values))
        else:
            quality[cell_id] = float(quality_signal[cell_id])
    return quality


def candidate_cells(
    cell_id: int,
    cell_count: int,
    pool_size: int,
    blocked: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.asarray(
        [
            candidate_id
            for candidate_id in range(cell_count)
            if candidate_id != cell_id and candidate_id not in blocked
        ],
        dtype=np.int64,
    )
    if len(candidates) <= pool_size:
        return candidates
    return rng.choice(candidates, size=pool_size, replace=False)


def validate_state_arrays(state_arrays: MutableMapping[str, np.ndarray]) -> int:
    if not state_arrays:
        raise ValueError("mobility state_arrays must not be empty")
    lengths = {name: len(values) for name, values in state_arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"mobility state arrays have mismatched lengths: {lengths}")
    return next(iter(unique_lengths))


def swap_cell_state(
    source_id: int,
    target_id: int,
    state_arrays: MutableMapping[str, np.ndarray],
    state_lists: MutableMapping[str, MutableSequence[object]] | None = None,
) -> None:
    for values in state_arrays.values():
        values[[source_id, target_id]] = values[[target_id, source_id]]
    if state_lists is None:
        return
    for values in state_lists.values():
        values[source_id], values[target_id] = values[target_id], values[source_id]


def apply_local_quality_mobility(
    state_arrays: MutableMapping[str, np.ndarray],
    quality_signal: np.ndarray,
    neighbors: list[list[int]],
    rng: np.random.Generator,
    params: MobilityParams | None = None,
    state_lists: MutableMapping[str, MutableSequence[object]] | None = None,
) -> MobilityStepResult:
    """Swap fixed-cell occupant state toward higher local-quality cells."""

    mobility_params = params or MobilityParams()
    cell_count = validate_state_arrays(state_arrays)
    if len(quality_signal) != cell_count:
        raise ValueError("mobility quality_signal length must match state arrays")
    if state_lists is not None:
        list_lengths = {name: len(values) for name, values in state_lists.items()}
        if any(length != cell_count for length in list_lengths.values()):
            raise ValueError(
                f"mobility state lists have mismatched lengths: {list_lengths}"
            )
    if not mobility_params.enabled or mobility_params.rate <= 0.0:
        return MobilityStepResult.none(cell_count)

    moved = np.zeros(cell_count, dtype=bool)
    targets = np.full(cell_count, -1, dtype=np.int64)
    gains = np.zeros(cell_count, dtype=np.float64)
    blocked: set[int] = set()
    quality = local_quality(quality_signal=quality_signal, neighbors=neighbors)

    for cell_id in rng.permutation(cell_count):
        source_id = int(cell_id)
        if source_id in blocked:
            continue
        if rng.random() >= mobility_params.rate:
            continue
        candidates = candidate_cells(
            cell_id=source_id,
            cell_count=cell_count,
            pool_size=mobility_params.candidate_pool_size,
            blocked=blocked,
            rng=rng,
        )
        if len(candidates) == 0:
            continue
        source_quality = float(quality[source_id])
        candidate_gains = (
            quality[candidates] - source_quality - mobility_params.move_cost
        )
        best_index = int(np.argmax(candidate_gains))
        best_gain = float(candidate_gains[best_index])
        if best_gain <= 0.0:
            continue
        target_id = int(candidates[best_index])

        swap_cell_state(
            source_id=source_id,
            target_id=target_id,
            state_arrays=state_arrays,
            state_lists=state_lists,
        )
        moved[source_id] = True
        targets[source_id] = target_id
        gains[source_id] = best_gain
        blocked.add(source_id)
        blocked.add(target_id)

    return MobilityStepResult(moved=moved, targets=targets, gains=gains)


def mobility_summary(result: MobilityStepResult) -> dict[str, float]:
    moved_gains = result.gains[result.moved]
    return {
        "mobility_rate": float(np.mean(result.moved)),
        "mean_mobility_gain": (
            float(np.mean(moved_gains)) if len(moved_gains) else 0.0
        ),
    }
