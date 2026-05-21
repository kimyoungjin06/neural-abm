"""Shared diagnostic plumbing for domain-owned binary learning traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from neural_abm.basin_phase_critic import (
    learned_basin_runtime_aggregate_fields,
    learned_basin_runtime_micro_fields,
)
from neural_abm.state_continuation import (
    basin_credit_diagnostics,
    basin_credit_micro_fields,
    basin_credit_training_diagnostics,
    basin_credit_training_micro_fields,
    domain_bootstrap_diagnostics,
    domain_bootstrap_micro_fields,
    domain_decision_bootstrap_diagnostics,
    domain_decision_bootstrap_micro_fields,
    domain_decision_replay_diagnostics,
    domain_decision_replay_micro_fields,
    domain_distill_bootstrap_diagnostics,
    domain_distill_bootstrap_micro_fields,
    domain_teacher_alignment_diagnostics,
    domain_teacher_alignment_micro_fields,
    state_continuation_diagnostics,
    state_continuation_micro_fields,
)


def _unique_field_names(*rows: Mapping[str, object]) -> list[str]:
    field_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                field_names.append(key)
                seen.add(key)
    return field_names


DOMAIN_LEARNING_AGGREGATE_FIELDS = _unique_field_names(
    state_continuation_diagnostics(None),
    basin_credit_diagnostics(None),
    basin_credit_training_diagnostics(
        diagnostics=None,
        training_components=None,
        actions=np.asarray([], dtype=np.int64),
    ),
    learned_basin_runtime_aggregate_fields(None),
    domain_bootstrap_diagnostics(None),
    domain_decision_bootstrap_diagnostics(None),
    domain_decision_replay_diagnostics(None),
    domain_distill_bootstrap_diagnostics(None),
    domain_teacher_alignment_diagnostics(None),
)


DOMAIN_LEARNING_MICRO_FIELDS = _unique_field_names(
    state_continuation_micro_fields(None, 0),
    basin_credit_micro_fields(None, 0),
    basin_credit_training_micro_fields(
        diagnostics=None,
        training_components=None,
        actions=np.asarray([], dtype=np.int64),
        agent_id=0,
    ),
    learned_basin_runtime_micro_fields(None, 0),
    domain_bootstrap_micro_fields(None, 0),
    domain_decision_bootstrap_micro_fields(None, 0),
    domain_decision_replay_micro_fields(None, 0),
    domain_distill_bootstrap_micro_fields(None, 0),
    domain_teacher_alignment_micro_fields(None, 0),
)


def domain_learning_aggregate_fields(
    *,
    extras: Mapping[str, Any],
    actions: np.ndarray,
) -> dict[str, object]:
    """Return common aggregate diagnostics carried in domain step extras.

    The domain still owns the meaning of the extra values. This helper only
    centralizes the stable CSV plumbing used by Toy2/Toy4 adapters.
    """

    action_values = np.asarray(actions, dtype=np.int64)
    return {
        **state_continuation_diagnostics(extras.get("state_continuation_components")),
        **basin_credit_diagnostics(extras.get("basin_credit_diagnostics")),
        **basin_credit_training_diagnostics(
            diagnostics=extras.get("basin_credit_diagnostics"),
            training_components=extras.get("basin_credit_training_components"),
            actions=action_values,
            training_action1_advantage=extras.get(
                "basin_credit_training_action1_advantage"
            ),
            training_credit_source=str(
                extras.get("basin_credit_training_credit_source", "prototype")
            ),
            training_replay_selection=str(
                extras.get("basin_credit_training_replay_selection", "all")
            ),
            training_replay_min_selected_rate=extras.get(
                "basin_credit_training_replay_min_selected_rate",
                "",
            ),
            training_replay_mask=extras.get("basin_credit_training_replay_mask"),
            training_replay_weight=extras.get("basin_credit_training_replay_weight"),
            learned_credit_used_mask=extras.get(
                "basin_credit_training_learned_credit_used_mask"
            ),
        ),
        **learned_basin_runtime_aggregate_fields(
            extras.get("basin_learned_diagnostics")
        ),
        **domain_bootstrap_diagnostics(extras.get("domain_bootstrap_diagnostics")),
        **domain_decision_bootstrap_diagnostics(
            extras.get("domain_decision_bootstrap_diagnostics")
        ),
        **domain_decision_replay_diagnostics(
            extras.get("domain_decision_replay_diagnostics")
        ),
        **domain_distill_bootstrap_diagnostics(
            extras.get("domain_distill_bootstrap_diagnostics")
        ),
        **domain_teacher_alignment_diagnostics(
            extras.get("domain_teacher_alignment_diagnostics")
        ),
    }


def domain_learning_micro_fields(
    *,
    extras: Mapping[str, Any],
    actions: np.ndarray,
    agent_id: int,
) -> dict[str, object]:
    """Return common per-agent diagnostics carried in domain step extras."""

    action_values = np.asarray(actions, dtype=np.int64)
    return {
        **state_continuation_micro_fields(
            extras.get("state_continuation_components"),
            agent_id,
        ),
        **basin_credit_micro_fields(
            extras.get("basin_credit_diagnostics"),
            agent_id,
        ),
        **basin_credit_training_micro_fields(
            diagnostics=extras.get("basin_credit_diagnostics"),
            training_components=extras.get("basin_credit_training_components"),
            actions=action_values,
            agent_id=agent_id,
            training_action1_advantage=extras.get(
                "basin_credit_training_action1_advantage"
            ),
            training_credit_source=str(
                extras.get("basin_credit_training_credit_source", "prototype")
            ),
            training_replay_selection=str(
                extras.get("basin_credit_training_replay_selection", "all")
            ),
            training_replay_mask=extras.get("basin_credit_training_replay_mask"),
            training_replay_weight=extras.get("basin_credit_training_replay_weight"),
            learned_credit_used_mask=extras.get(
                "basin_credit_training_learned_credit_used_mask"
            ),
        ),
        **learned_basin_runtime_micro_fields(
            extras.get("basin_learned_diagnostics"),
            agent_id,
        ),
        **domain_bootstrap_micro_fields(
            extras.get("domain_bootstrap_diagnostics"),
            agent_id,
        ),
        **domain_decision_bootstrap_micro_fields(
            extras.get("domain_decision_bootstrap_diagnostics"),
            agent_id,
        ),
        **domain_decision_replay_micro_fields(
            extras.get("domain_decision_replay_diagnostics"),
            agent_id,
        ),
        **domain_distill_bootstrap_micro_fields(
            extras.get("domain_distill_bootstrap_diagnostics"),
            agent_id,
        ),
        **domain_teacher_alignment_micro_fields(
            extras.get("domain_teacher_alignment_diagnostics"),
            agent_id,
        ),
    }
