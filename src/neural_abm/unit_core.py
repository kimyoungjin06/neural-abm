"""Torch-free lifecycle reports and diagnostics for lightweight API profiles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from neural_abm.social_core import SocialMixResult


@dataclass(frozen=True)
class CommitReport:
    """Result of committing a social mix into concrete agent state."""

    channel: str
    commit_mode: str
    committed_agent_ids: list[int]
    losses: list[float]
    update_norms: list[float]

    @classmethod
    def from_mix_result(
        cls,
        mix_result: SocialMixResult,
        committed_agent_ids: list[int] | None = None,
        losses: list[float] | None = None,
    ) -> "CommitReport":
        return cls(
            channel=mix_result.channel,
            commit_mode=mix_result.commit_mode,
            committed_agent_ids=committed_agent_ids or [],
            losses=losses if losses is not None else list(mix_result.losses),
            update_norms=list(mix_result.update_norms),
        )


@dataclass(frozen=True)
class SocialDiagnostics:
    """Flat diagnostics derived from one social mix result."""

    channel: str
    commit_mode: str
    peer_counts: list[int]
    losses: list[float]
    update_norms: list[float]
    active_agent_count: int
    mean_peer_count: float
    mean_loss: float
    mean_update_norm: float
    max_update_norm: float

    def micro_row(self, agent_id: int) -> dict[str, Any]:
        return {
            "social_channel": self.channel,
            "commit_mode": self.commit_mode,
            "social_loss": self.losses[agent_id],
            "social_update_norm": self.update_norms[agent_id],
        }

    def aggregate_row(self) -> dict[str, Any]:
        return {
            "social_channel": self.channel,
            "commit_mode": self.commit_mode,
            "mean_social_loss": self.mean_loss,
            "mean_social_update_norm": self.mean_update_norm,
            "max_social_update_norm": self.max_update_norm,
            "active_social_agent_count": self.active_agent_count,
        }


def social_diagnostics(mix_result: SocialMixResult) -> SocialDiagnostics:
    """Build reusable diagnostics from a social mix result."""

    peer_counts = [len(peers) for peers in mix_result.peer_ids]
    losses = list(mix_result.losses)
    update_norms = list(mix_result.update_norms)
    return SocialDiagnostics(
        channel=mix_result.channel,
        commit_mode=mix_result.commit_mode,
        peer_counts=peer_counts,
        losses=losses,
        update_norms=update_norms,
        active_agent_count=sum(1 for count in peer_counts if count > 0),
        mean_peer_count=float(sum(peer_counts) / len(peer_counts)) if peer_counts else 0.0,
        mean_loss=float(sum(losses) / len(losses)) if losses else 0.0,
        mean_update_norm=(
            float(sum(update_norms) / len(update_norms)) if update_norms else 0.0
        ),
        max_update_norm=max(update_norms, default=0.0),
    )


class CommitAdapter(Protocol):
    """Adapter that commits mixed social values back into agents."""

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        """Apply a social mix result and return commit diagnostics."""


@dataclass(frozen=True)
class LocalUpdateReport:
    """Result of committing one local learning step."""

    losses: Any
    active_agent_ids: list[int] | None = None
    update_result: Any | None = None
    diagnostics: Mapping[str, Any] | None = None


class LocalUpdateAdapter(Protocol):
    """Adapter that commits domain-local learning into agent/backend state."""

    def update(self, *args: Any, **kwargs: Any) -> LocalUpdateReport:
        """Apply local learning and return update diagnostics."""


@dataclass
class NABMLocalStep:
    """Reusable lifecycle unit for local learning updates."""

    update_adapter: LocalUpdateAdapter

    def run(self, *args: Any, **kwargs: Any) -> LocalUpdateReport:
        return self.update_adapter.update(*args, **kwargs)


@dataclass(frozen=True)
class NABMStepResult:
    """Full result of one reusable NABM social step."""

    mix: SocialMixResult
    commit: CommitReport
    diagnostics: SocialDiagnostics


PeerSelector = Callable[[Sequence[Mapping[str, Any]]], list[list[int]]]
SocialValueBuilder = Callable[
    [Sequence[Any], Sequence[Mapping[str, Any]]],
    Any,
]


__all__ = [
    "CommitAdapter",
    "CommitReport",
    "LocalUpdateAdapter",
    "LocalUpdateReport",
    "NABMLocalStep",
    "NABMStepResult",
    "PeerSelector",
    "SocialDiagnostics",
    "SocialValueBuilder",
    "social_diagnostics",
]
