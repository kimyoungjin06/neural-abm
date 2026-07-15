"""Torch-free workflow scaffolds for narrow researcher-facing examples."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_abm.social_core import (
    BOUNDED_SCALAR_CHANNEL,
    PeerSelectionResult,
    SocialChannel,
    SocialMixResult,
    mix_bounded_scalars,
    select_bounded_scalar_output_peers,
)
from neural_abm.unit_core import CommitReport, SocialDiagnostics, social_diagnostics

WORKFLOW_STEPS: tuple[str, ...] = (
    "define_domain_state",
    "apply_local_adaptation",
    "declare_typed_peer_exchange",
    "commit_domain_transition",
    "audit_aggregate_and_micro_evidence",
)

LocalUpdate = Callable[[Any], float]
DomainTransition = Callable[[Any, float], Mapping[str, Any]]
MicroFields = Callable[[Any], Mapping[str, Any]]
AggregateFields = Callable[[Sequence[Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class BoundedScalarWorkflowSpec:
    """Minimal settings for an auditable bounded-scalar workflow.

    The spec owns generic workflow settings only. Domain meaning remains in the
    user's agent fields, local update callback, and domain transition callback.
    """

    domain_question: str
    state_field: str
    channel_name: str
    transition_label: str
    commit_mode: str
    peer_rule: str = "output_similarity"
    peer_similarity_threshold: float = 0.70
    social_alpha: float = 0.45
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    round_digits: int | None = 4

    def __post_init__(self) -> None:
        if not self.domain_question:
            raise ValueError("domain_question must be non-empty")
        if not self.state_field:
            raise ValueError("state_field must be non-empty")
        if not self.channel_name:
            raise ValueError("channel_name must be non-empty")
        if not self.transition_label:
            raise ValueError("transition_label must be non-empty")
        if not self.commit_mode:
            raise ValueError("commit_mode must be non-empty")
        if not 0.0 <= self.peer_similarity_threshold <= 1.0:
            raise ValueError("peer_similarity_threshold must lie in [0, 1]")
        if not 0.0 <= self.social_alpha <= 1.0:
            raise ValueError("social_alpha must lie in [0, 1]")
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
            raise ValueError("bounds must be finite")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound")


@dataclass(frozen=True)
class BoundedScalarWorkflowResult:
    """Result envelope for a bounded-scalar researcher workflow."""

    spec: BoundedScalarWorkflowSpec
    peer_selection: PeerSelectionResult
    mix_result: SocialMixResult
    commit_report: CommitReport
    diagnostics: SocialDiagnostics
    aggregate_audit: dict[str, Any]
    micro_audit: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "surface": "neural_abm.workflow_lite",
            "base_surface": "neural_abm.api_lite",
            "default_profile": "torch-free",
            "workflow": list(WORKFLOW_STEPS),
            "domain_question": self.spec.domain_question,
            "domain_owned": {
                "state": [self.spec.state_field],
                "transition": self.spec.transition_label,
            },
            "unit_standardized": {
                "channel": self.commit_report.channel,
                "commit_mode": self.commit_report.commit_mode,
                "peer_rule": self.spec.peer_rule,
                "peer_similarity_threshold": self.spec.peer_similarity_threshold,
                "social_alpha": self.spec.social_alpha,
            },
            "aggregate_audit": self.aggregate_audit,
            "micro_audit": self.micro_audit,
        }


def run_bounded_scalar_workflow(
    *,
    agents: Sequence[Any],
    neighbors: list[list[int]],
    spec: BoundedScalarWorkflowSpec,
    local_update: LocalUpdate,
    domain_transition: DomainTransition,
    micro_fields: MicroFields | None = None,
    aggregate_fields: AggregateFields | None = None,
) -> BoundedScalarWorkflowResult:
    """Run a narrow bounded-scalar workflow and return auditable rows."""

    local_losses = [float(local_update(agent)) for agent in agents]
    values_after_local = np.asarray(
        [_bounded_state_value(agent, spec.state_field) for agent in agents],
        dtype=np.float64,
    )
    channel = SocialChannel(
        name=spec.channel_name,
        kind=BOUNDED_SCALAR_CHANNEL,
        commit_mode=spec.commit_mode,
        lower_bound=spec.lower_bound,
        upper_bound=spec.upper_bound,
    )
    peer_selection = select_bounded_scalar_output_peers(
        neighbors=neighbors,
        values=values_after_local,
        peer_rule=spec.peer_rule,
        threshold=spec.peer_similarity_threshold,
        lower_bound=spec.lower_bound,
        upper_bound=spec.upper_bound,
    )
    mix_result = mix_bounded_scalars(
        values=values_after_local,
        peer_ids=peer_selection.peer_ids,
        alpha=spec.social_alpha,
        lower_bound=spec.lower_bound,
        upper_bound=spec.upper_bound,
        channel=channel.name,
        commit_mode=channel.commit_mode,
    )
    commit_report = CommitReport.from_mix_result(
        mix_result,
        committed_agent_ids=[
            _agent_id(agent, index) for index, agent in enumerate(agents)
        ],
    )
    transition_rows = [
        dict(domain_transition(agent, float(value)))
        for agent, value in zip(agents, mix_result.mixed_values, strict=True)
    ]
    diagnostics = social_diagnostics(mix_result)
    micro_audit = _micro_audit_rows(
        agents=agents,
        spec=spec,
        local_losses=local_losses,
        transition_rows=transition_rows,
        diagnostics=diagnostics,
        micro_fields=micro_fields,
    )
    aggregate_audit = _aggregate_audit_row(
        agents=agents,
        spec=spec,
        local_losses=local_losses,
        diagnostics=diagnostics,
        aggregate_fields=aggregate_fields,
    )
    return BoundedScalarWorkflowResult(
        spec=spec,
        peer_selection=peer_selection,
        mix_result=mix_result,
        commit_report=commit_report,
        diagnostics=diagnostics,
        aggregate_audit=aggregate_audit,
        micro_audit=micro_audit,
    )


def _agent_id(agent: Any, fallback: int) -> int:
    if isinstance(agent, Mapping) and "agent_id" in agent:
        return int(agent["agent_id"])
    if hasattr(agent, "agent_id"):
        return int(getattr(agent, "agent_id"))
    return fallback


def _bounded_state_value(agent: Any, state_field: str) -> float:
    if isinstance(agent, Mapping):
        value = agent[state_field]
    else:
        value = getattr(agent, state_field)
    return float(value)


def _round_value(value: Any, digits: int | None) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if digits is None else round(numeric, digits)
    return value


def _micro_audit_rows(
    *,
    agents: Sequence[Any],
    spec: BoundedScalarWorkflowSpec,
    local_losses: Sequence[float],
    transition_rows: Sequence[Mapping[str, Any]],
    diagnostics: SocialDiagnostics,
    micro_fields: MicroFields | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, agent in enumerate(agents):
        row = {
            "agent_id": _agent_id(agent, index),
            spec.state_field: _round_value(
                _bounded_state_value(agent, spec.state_field),
                spec.round_digits,
            ),
            "peer_count": diagnostics.peer_counts[index],
            "local_shift": _round_value(local_losses[index], spec.round_digits),
            "social_shift": _round_value(
                diagnostics.update_norms[index],
                spec.round_digits,
            ),
        }
        if micro_fields is not None:
            row.update(
                {
                    key: _round_value(value, spec.round_digits)
                    for key, value in micro_fields(agent).items()
                }
            )
        row.update(
            {
                key: _round_value(value, spec.round_digits)
                for key, value in transition_rows[index].items()
            }
        )
        rows.append(row)
    return rows


def _aggregate_audit_row(
    *,
    agents: Sequence[Any],
    spec: BoundedScalarWorkflowSpec,
    local_losses: Sequence[float],
    diagnostics: SocialDiagnostics,
    aggregate_fields: AggregateFields | None,
) -> dict[str, Any]:
    state_values = [_bounded_state_value(agent, spec.state_field) for agent in agents]
    row: dict[str, Any] = {
        "agent_count": len(agents),
        f"mean_{spec.state_field}": _round_value(
            float(np.mean(state_values)) if state_values else 0.0,
            spec.round_digits,
        ),
        "mean_local_shift": _round_value(
            float(np.mean(local_losses)) if local_losses else 0.0,
            spec.round_digits,
        ),
        "mean_peer_count": _round_value(
            diagnostics.mean_peer_count,
            spec.round_digits,
        ),
        "mean_social_shift": _round_value(
            diagnostics.mean_update_norm,
            spec.round_digits,
        ),
        "max_social_shift": _round_value(
            diagnostics.max_update_norm,
            spec.round_digits,
        ),
    }
    if aggregate_fields is not None:
        row.update(
            {
                key: _round_value(value, spec.round_digits)
                for key, value in aggregate_fields(agents).items()
            }
        )
    return row


__all__ = [
    "AggregateFields",
    "BoundedScalarWorkflowResult",
    "BoundedScalarWorkflowSpec",
    "DomainTransition",
    "LocalUpdate",
    "MicroFields",
    "WORKFLOW_STEPS",
    "run_bounded_scalar_workflow",
]
