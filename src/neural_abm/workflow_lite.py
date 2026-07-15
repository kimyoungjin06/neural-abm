"""Torch-free workflow scaffolds for narrow researcher-facing examples."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
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

MICRO_AUDIT_RESERVED_FIELDS: frozenset[str] = frozenset(
    {
        "agent_id",
        "peer_count",
        "local_shift",
        "social_shift",
    }
)
AGGREGATE_AUDIT_RESERVED_FIELDS: frozenset[str] = frozenset(
    {
        "agent_count",
        "mean_local_shift",
        "mean_peer_count",
        "mean_social_shift",
        "max_social_shift",
    }
)


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
        if self.state_field in MICRO_AUDIT_RESERVED_FIELDS:
            raise ValueError(
                f"state_field is reserved by the workflow audit contract: "
                f"{self.state_field}"
            )
        if not 0.0 <= self.peer_similarity_threshold <= 1.0:
            raise ValueError("peer_similarity_threshold must lie in [0, 1]")
        if not 0.0 <= self.social_alpha <= 1.0:
            raise ValueError("social_alpha must lie in [0, 1]")
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
            raise ValueError("bounds must be finite")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be <= upper_bound")
        if self.round_digits is not None and (
            isinstance(self.round_digits, bool)
            or not isinstance(self.round_digits, Integral)
        ):
            raise ValueError("round_digits must be an integer or None")


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

    def to_dict(self, *, round_values: bool = True) -> dict[str, Any]:
        payload = {
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
        if not round_values:
            return payload
        serialized = dict(payload)
        serialized["aggregate_audit"] = _serialize_value(
            self.aggregate_audit,
            self.spec.round_digits,
        )
        serialized["micro_audit"] = _serialize_value(
            self.micro_audit,
            self.spec.round_digits,
        )
        return serialized


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


def _serialize_value(value: Any, digits: int | None) -> Any:
    if isinstance(value, Mapping):
        return {key: _serialize_value(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item, digits) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_value(item, digits) for item in value)
    return _round_value(value, digits)


def _reject_reserved_fields(
    fields: Mapping[str, Any],
    reserved: set[str] | frozenset[str],
    *,
    source: str,
) -> None:
    collisions = sorted(set(fields).intersection(reserved))
    if collisions:
        joined = ", ".join(collisions)
        raise ValueError(f"{source} cannot overwrite reserved audit fields: {joined}")


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
    reserved = set(MICRO_AUDIT_RESERVED_FIELDS)
    reserved.add(spec.state_field)
    for index, agent in enumerate(agents):
        row = {
            "agent_id": _agent_id(agent, index),
            spec.state_field: _bounded_state_value(agent, spec.state_field),
            "peer_count": diagnostics.peer_counts[index],
            "local_shift": float(local_losses[index]),
            "social_shift": float(diagnostics.update_norms[index]),
        }
        if micro_fields is not None:
            domain_fields = dict(micro_fields(agent))
            _reject_reserved_fields(
                domain_fields,
                reserved,
                source="micro_fields",
            )
            row.update(
                {key: _round_value(value, None) for key, value in domain_fields.items()}
            )
        transition_fields = dict(transition_rows[index])
        _reject_reserved_fields(
            transition_fields,
            reserved,
            source="domain_transition",
        )
        row.update(
            {key: _round_value(value, None) for key, value in transition_fields.items()}
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
        f"mean_{spec.state_field}": (
            float(np.mean(state_values)) if state_values else 0.0
        ),
        "mean_local_shift": (float(np.mean(local_losses)) if local_losses else 0.0),
        "mean_peer_count": float(diagnostics.mean_peer_count),
        "mean_social_shift": float(diagnostics.mean_update_norm),
        "max_social_shift": float(diagnostics.max_update_norm),
    }
    if aggregate_fields is not None:
        domain_fields = dict(aggregate_fields(agents))
        reserved = set(AGGREGATE_AUDIT_RESERVED_FIELDS)
        reserved.add(f"mean_{spec.state_field}")
        _reject_reserved_fields(
            domain_fields,
            reserved,
            source="aggregate_fields",
        )
        row.update(
            {key: _round_value(value, None) for key, value in domain_fields.items()}
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
