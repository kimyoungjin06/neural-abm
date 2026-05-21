"""Dataset artifacts for basin-transition critic training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neural_abm.state_continuation import (
    BasinCreditDiagnostics,
    StateContinuationComponents,
    build_basin_phase_representation,
    selected_credit_to_action1_advantage,
)


BASIN_PHASE_FIELD_NAMES: tuple[str, ...] = (
    "phase_payoff_alignment",
    "phase_action_rate",
    "phase_policy_rate",
    "phase_consensus",
    "phase_payoff_stability",
)

DEFAULT_FUTURE_BASIN_HORIZON = 5

FUTURE_BASIN_MOTION_FIELDS: tuple[str, ...] = (
    "future_basin_horizon",
    "future_mean_payoff",
    "future_basin_score_delta",
    "future_ceiling_reached",
    "future_epochs_to_ceiling",
    "future_basin_motion_positive",
)

BASIN_TRANSITION_SAMPLE_FIELDS: tuple[str, ...] = (
    "sample_schema_version",
    "toy",
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "agent_count",
    "target_basin",
    "critic",
    "credit_method",
    "objective_profile",
    "objective_weight",
    "individual_weight",
    "local_social_weight",
    "basin_weight",
    "training_scope",
    "training_pass_schedule",
    "training_passes",
    "configured_training_passes",
    "min_training_passes",
    "training_pass_score_threshold",
    "training_pass_credit_positive_threshold",
    "training_pass_credit_delta_threshold",
    "action_observed",
    "action_counterfactual",
    "policy_action_probability",
    "mean_payoff",
    "target_payoff",
    "action_rate",
    "policy_rate",
    *BASIN_PHASE_FIELD_NAMES,
    "score_observed",
    "score_counterfactual",
    "selected_action_credit",
    "score_delta",
    "credit_positive",
    "phase_confidence",
    "material_advantage",
    "local_social_advantage",
    "objective_effective_advantage",
    "basin_action1_advantage",
    "training_basin_action1_advantage",
    "training_credit_source",
    "training_replay_selection",
    "training_replay_min_selected_rate",
    "training_replay_selected",
    "training_replay_weight",
    "learned_credit_used",
    "training_effective_advantage",
    *FUTURE_BASIN_MOTION_FIELDS,
    "final_mean_payoff",
    "final_ceiling_reached",
    "time_to_ceiling",
)


def _vector(values: np.ndarray, *, name: str, dtype: object) -> np.ndarray:
    vector = np.asarray(values, dtype=dtype)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return vector


def _float(value: object) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _domain_fields(domain_fields: Mapping[str, object] | None) -> dict[str, object]:
    if domain_fields is None:
        return {}
    return dict(domain_fields)


def basin_transition_sample_rows(
    *,
    toy: str,
    run_id: str,
    seed: int,
    epoch: int,
    actions: np.ndarray,
    payoffs: np.ndarray,
    action_probabilities: np.ndarray,
    target_payoff: float,
    diagnostics: BasinCreditDiagnostics | None,
    objective_components: StateContinuationComponents | None,
    training_components: StateContinuationComponents | None,
    training_action1_advantage: np.ndarray | None = None,
    training_credit_source: str = "prototype",
    training_replay_selection: str = "all",
    training_replay_min_selected_rate: float | str = "",
    training_replay_mask: np.ndarray | None = None,
    training_replay_weight: np.ndarray | None = None,
    learned_credit_used_mask: np.ndarray | None = None,
    domain_fields: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Return per-agent counterfactual basin-transition sample rows."""

    if (
        diagnostics is None
        or objective_components is None
        or training_components is None
    ):
        return []

    action_values = _vector(actions, name="actions", dtype=np.int64)
    payoff_values = _vector(payoffs, name="payoffs", dtype=np.float64)
    probability_values = _vector(
        action_probabilities,
        name="action_probabilities",
        dtype=np.float64,
    )
    if not (
        len(action_values) == len(payoff_values) == len(probability_values)
    ):
        raise ValueError("actions, payoffs, and action probabilities must match")

    embedding = build_basin_phase_representation(
        actions=action_values,
        payoffs=payoff_values,
        target_payoff=float(target_payoff),
        action_probabilities=probability_values,
    )[0]
    phase_fields = dict(zip(BASIN_PHASE_FIELD_NAMES, map(float, embedding), strict=True))
    mean_payoff = float(np.mean(payoff_values))
    action_rate = float(np.mean(action_values))
    policy_rate = float(np.mean(probability_values))
    basin_action1_advantage = selected_credit_to_action1_advantage(
        diagnostics.selected_action_credit,
        action_values,
    )
    training_basin_action1_advantage = (
        basin_action1_advantage
        if training_action1_advantage is None
        else _vector(
            training_action1_advantage,
            name="training_action1_advantage",
            dtype=np.float64,
        )
    )
    if training_basin_action1_advantage.shape != basin_action1_advantage.shape:
        raise ValueError(
            "training_action1_advantage shape "
            f"{training_basin_action1_advantage.shape} must match basin advantage "
            f"shape {basin_action1_advantage.shape}"
        )
    learned_credit_used = (
        np.zeros(basin_action1_advantage.shape, dtype=bool)
        if learned_credit_used_mask is None
        else _vector(
            learned_credit_used_mask,
            name="learned_credit_used_mask",
            dtype=bool,
        )
    )
    if learned_credit_used.shape != basin_action1_advantage.shape:
        raise ValueError(
            "learned_credit_used_mask shape "
            f"{learned_credit_used.shape} must match basin advantage shape "
            f"{basin_action1_advantage.shape}"
        )
    training_replay_selected = (
        diagnostics.applied_mask.copy()
        if training_replay_mask is None
        else _vector(
            training_replay_mask,
            name="training_replay_mask",
            dtype=bool,
        )
    )
    if training_replay_selected.shape != basin_action1_advantage.shape:
        raise ValueError(
            "training_replay_mask shape "
            f"{training_replay_selected.shape} must match basin advantage shape "
            f"{basin_action1_advantage.shape}"
        )
    training_replay_weight_values = (
        training_replay_selected.astype(np.float64)
        if training_replay_weight is None
        else _vector(
            training_replay_weight,
            name="training_replay_weight",
            dtype=np.float64,
        )
    )
    if training_replay_weight_values.shape != basin_action1_advantage.shape:
        raise ValueError(
            "training_replay_weight shape "
            f"{training_replay_weight_values.shape} must match basin advantage shape "
            f"{basin_action1_advantage.shape}"
        )
    training_replay_weight_values = np.where(
        training_replay_selected,
        np.clip(training_replay_weight_values, 0.0, 1.0),
        0.0,
    )
    shared_fields = {
        "sample_schema_version": 1,
        "toy": toy,
        "run_id": run_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "agent_count": len(action_values),
        "target_basin": diagnostics.target,
        "critic": diagnostics.critic,
        "credit_method": diagnostics.credit_method,
        "objective_profile": objective_components.objective_profile,
        "objective_weight": diagnostics.objective_weight,
        "individual_weight": diagnostics.individual_weight,
        "local_social_weight": diagnostics.local_social_weight,
        "basin_weight": diagnostics.weight,
        "training_scope": diagnostics.training_scope,
        "training_pass_schedule": diagnostics.training_pass_schedule,
        "training_passes": diagnostics.training_passes,
        "configured_training_passes": diagnostics.configured_training_passes,
        "min_training_passes": diagnostics.min_training_passes,
        "training_pass_score_threshold": diagnostics.training_pass_score_threshold,
        "training_pass_credit_positive_threshold": (
            diagnostics.training_pass_credit_positive_threshold
        ),
        "training_pass_credit_delta_threshold": (
            diagnostics.training_pass_credit_delta_threshold
        ),
        "mean_payoff": mean_payoff,
        "target_payoff": float(target_payoff),
        "action_rate": action_rate,
        "policy_rate": policy_rate,
        **phase_fields,
        **_domain_fields(domain_fields),
    }

    rows: list[dict[str, object]] = []
    for agent_id in np.flatnonzero(diagnostics.applied_mask):
        agent_index = int(agent_id)
        selected_credit = _float(diagnostics.selected_action_credit[agent_index])
        observed_action = int(action_values[agent_index])
        rows.append(
            {
                **shared_fields,
                "agent_id": agent_index,
                "action_observed": observed_action,
                "action_counterfactual": 1 - observed_action,
                "policy_action_probability": _float(probability_values[agent_index]),
                "score_observed": _float(diagnostics.score_observed[agent_index]),
                "score_counterfactual": _float(
                    diagnostics.score_counterfactual[agent_index]
                ),
                "selected_action_credit": selected_credit,
                "score_delta": selected_credit,
                "credit_positive": bool(selected_credit > 0.0),
                "phase_confidence": _float(diagnostics.phase_confidence[agent_index]),
                "material_advantage": _float(
                    objective_components.material[agent_index]
                ),
                "local_social_advantage": _float(
                    objective_components.social[agent_index]
                ),
                "objective_effective_advantage": _float(
                    objective_components.effective[agent_index]
                ),
                "basin_action1_advantage": _float(
                    basin_action1_advantage[agent_index]
                ),
                "training_basin_action1_advantage": _float(
                    training_basin_action1_advantage[agent_index]
                ),
                "training_credit_source": training_credit_source,
                "training_replay_selection": training_replay_selection,
                "training_replay_min_selected_rate": (
                    training_replay_min_selected_rate
                ),
                "training_replay_selected": bool(
                    training_replay_selected[agent_index]
                ),
                "training_replay_weight": float(
                    training_replay_weight_values[agent_index]
                ),
                "learned_credit_used": bool(learned_credit_used[agent_index]),
                "training_effective_advantage": _float(
                    training_components.effective[agent_index]
                ),
                "future_basin_horizon": None,
                "future_mean_payoff": None,
                "future_basin_score_delta": None,
                "future_ceiling_reached": None,
                "future_epochs_to_ceiling": None,
                "future_basin_motion_positive": None,
                "final_mean_payoff": None,
                "final_ceiling_reached": None,
                "time_to_ceiling": None,
            }
        )
    return rows


def annotate_terminal_outcomes(
    samples: Sequence[Mapping[str, object]],
    *,
    final_mean_payoff: float,
    target_payoff: float,
    ceiling_tolerance: float = 1e-12,
    future_horizon: int = DEFAULT_FUTURE_BASIN_HORIZON,
) -> list[dict[str, object]]:
    """Attach run-terminal labels to transition sample rows."""

    rows = [dict(row) for row in samples]
    if not rows:
        return rows
    threshold = float(target_payoff) - float(ceiling_tolerance)
    time_to_ceiling = min(
        (
            int(row["epoch"])
            for row in rows
            if float(row["mean_payoff"]) >= threshold
        ),
        default=None,
    )
    final_ceiling_reached = float(final_mean_payoff) >= threshold
    for row in rows:
        row["final_mean_payoff"] = float(final_mean_payoff)
        row["final_ceiling_reached"] = final_ceiling_reached
        row["time_to_ceiling"] = time_to_ceiling
    frame = pd.DataFrame(rows)
    frame = ensure_future_basin_motion_labels(
        frame,
        future_horizon=future_horizon,
        ceiling_tolerance=ceiling_tolerance,
    )
    return frame.to_dict(orient="records")


def ensure_future_basin_motion_labels(
    samples: pd.DataFrame,
    *,
    future_horizon: int = DEFAULT_FUTURE_BASIN_HORIZON,
    ceiling_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return samples with forward basin-motion labels filled per run."""

    required = {"epoch", "mean_payoff", "target_payoff"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(
            "Basin transition samples missing field(s): " + ", ".join(missing)
        )

    frame = samples.copy()
    horizon = max(int(future_horizon), 0)
    for field in (
        "future_mean_payoff",
        "future_basin_score_delta",
        "future_epochs_to_ceiling",
    ):
        if field not in frame.columns:
            frame[field] = np.nan
    for field in ("future_ceiling_reached", "future_basin_motion_positive"):
        if field not in frame.columns:
            frame[field] = pd.Series([None] * len(frame), index=frame.index)
        else:
            frame[field] = frame[field].astype(object)
    frame["future_basin_horizon"] = horizon

    group_keys = [
        key
        for key in ("source_path", "run_id", "seed")
        if key in frame.columns
    ]
    if group_keys:
        groups = frame.groupby(group_keys, sort=False, dropna=False)
        group_indices = (group.index for _, group in groups)
    else:
        group_indices = (frame.index,)

    for index in group_indices:
        group = frame.loc[index]
        epoch = pd.to_numeric(group["epoch"], errors="coerce")
        mean_payoff = pd.to_numeric(group["mean_payoff"], errors="coerce")
        target_payoff = pd.to_numeric(group["target_payoff"], errors="coerce")
        epoch_table = (
            pd.DataFrame(
                {
                    "epoch": epoch,
                    "mean_payoff": mean_payoff,
                    "target_payoff": target_payoff,
                },
                index=group.index,
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["epoch", "mean_payoff", "target_payoff"])
        )
        if epoch_table.empty:
            continue
        by_epoch = (
            epoch_table.groupby("epoch", sort=True)
            .agg(mean_payoff=("mean_payoff", "mean"), target_payoff=("target_payoff", "mean"))
            .reset_index()
        )
        epoch_values = by_epoch["epoch"].to_numpy(dtype=np.float64)
        payoff_values = by_epoch["mean_payoff"].to_numpy(dtype=np.float64)
        target_values = by_epoch["target_payoff"].to_numpy(dtype=np.float64)
        thresholds = target_values - float(ceiling_tolerance)
        reached_by_epoch = payoff_values >= thresholds

        for row_index, row in epoch_table.iterrows():
            current_epoch = float(row["epoch"])
            current_payoff = float(row["mean_payoff"])
            window = (epoch_values >= current_epoch) & (
                epoch_values <= current_epoch + horizon
            )
            if not np.any(window):
                future_epoch = current_epoch
                future_payoff = current_payoff
            else:
                future_position = int(np.flatnonzero(window)[-1])
                future_epoch = float(epoch_values[future_position])
                future_payoff = float(payoff_values[future_position])
            ceiling_window = window & reached_by_epoch
            if np.any(ceiling_window):
                ceiling_epoch = float(epoch_values[int(np.flatnonzero(ceiling_window)[0])])
                future_ceiling_reached = True
                future_epochs_to_ceiling = ceiling_epoch - current_epoch
            else:
                future_ceiling_reached = False
                future_epochs_to_ceiling = np.nan
            delta = future_payoff - current_payoff
            frame.loc[row_index, "future_mean_payoff"] = future_payoff
            frame.loc[row_index, "future_basin_score_delta"] = delta
            frame.loc[row_index, "future_ceiling_reached"] = future_ceiling_reached
            frame.loc[row_index, "future_epochs_to_ceiling"] = (
                future_epochs_to_ceiling
            )
            frame.loc[row_index, "future_basin_motion_positive"] = delta > 0.0
            if future_epoch == current_epoch and horizon == 0:
                frame.loc[row_index, "future_epochs_to_ceiling"] = (
                    0.0 if future_ceiling_reached else np.nan
                )

    return frame


def write_basin_transition_samples(
    run_dir: Path,
    samples: Sequence[Mapping[str, object]],
) -> Path | None:
    """Write basin transition samples as a Parquet artifact."""

    if not samples:
        return None
    path = run_dir / "basin_transition_samples.parquet"
    frame = pd.DataFrame(list(samples))
    extra_columns = sorted(
        column
        for column in frame.columns
        if column not in BASIN_TRANSITION_SAMPLE_FIELDS
    )
    frame = frame.reindex(columns=[*BASIN_TRANSITION_SAMPLE_FIELDS, *extra_columns])
    frame.to_parquet(path, index=False)
    return path
