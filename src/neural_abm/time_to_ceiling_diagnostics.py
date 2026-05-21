"""Time-to-ceiling bottleneck diagnostics for basin-credit evidence runs."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPOCH_DIAGNOSTIC_FIELDS = [
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "epoch",
    "time_to_ceiling",
    "final_within_ceiling",
    "mean_payoff",
    "action_rate",
    "mean_policy_action_probability_pre_revision",
    "mean_policy_action_probability_post_local",
    "mean_policy_action_probability_post_social",
    "policy_action_probability_pre_revision_gt_0p5_rate",
    "policy_action_probability_pre_revision_gt_0p7_rate",
    "policy_action_probability_pre_revision_gt_0p9_rate",
    "policy_action_probability_pre_revision_dwell_0p4_0p6_rate",
    "policy_action_probability_pre_revision_p10",
    "policy_action_probability_pre_revision_p50",
    "policy_action_probability_pre_revision_p90",
    "policy_action_probability_post_local_gt_0p5_rate",
    "policy_action_probability_post_local_gt_0p7_rate",
    "policy_action_probability_post_local_gt_0p9_rate",
    "policy_action_probability_post_local_dwell_0p4_0p6_rate",
    "policy_action_probability_post_local_p10",
    "policy_action_probability_post_local_p50",
    "policy_action_probability_post_local_p90",
    "policy_action_probability_post_social_gt_0p5_rate",
    "policy_action_probability_post_social_gt_0p7_rate",
    "policy_action_probability_post_social_gt_0p9_rate",
    "policy_action_probability_post_social_dwell_0p4_0p6_rate",
    "policy_action_probability_post_social_p10",
    "policy_action_probability_post_social_p50",
    "policy_action_probability_post_social_p90",
    "policy_probability_threshold_crossings_0p5_count",
    "policy_probability_threshold_crossings_0p7_count",
    "policy_probability_threshold_crossings_0p9_count",
    "policy_probability_threshold_crossings_0p5_rate",
    "policy_probability_threshold_crossings_0p7_rate",
    "policy_probability_threshold_crossings_0p9_rate",
    "action_flip_count",
    "action_flip_rate",
    "policy_pre_to_local_delta",
    "policy_local_to_social_delta",
    "policy_social_to_action_gap",
    "action_rate_delta",
    "mean_payoff_delta",
    "realized_revision_rate",
    "mean_local_loss",
    "mean_social_loss",
    "domain_effective_advantage_mean",
    "domain_effective_advantage_positive_rate",
    "domain_basin_action1_advantage_mean",
    "domain_basin_action1_advantage_positive_rate",
    "domain_basin_training_effective_advantage_mean",
    "domain_basin_training_effective_advantage_positive_rate",
    "domain_basin_training_effective_advantage_abs_mean",
    "domain_basin_credit_positive_rate",
    "domain_basin_score_delta_mean",
    "domain_resource_fraction",
]


RUN_DIAGNOSTIC_FIELDS = [
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "time_to_ceiling",
    "final_within_ceiling",
    "ever_reached_ceiling",
    "window_epoch_count",
    "first_positive_training_advantage_epoch",
    "first_local_policy_increase_epoch",
    "first_social_nonnegative_epoch",
    "first_action_rate_increase_epoch",
    "early_effective_advantage_mean",
    "early_effective_advantage_positive_rate",
    "early_basin_action1_advantage_mean",
    "early_basin_action1_advantage_positive_rate",
    "early_training_effective_advantage_mean",
    "early_training_effective_advantage_positive_rate",
    "early_training_effective_advantage_abs_mean",
    "early_policy_pre_to_local_delta_mean",
    "early_policy_local_to_social_delta_mean",
    "early_policy_social_to_action_gap_mean",
    "early_action_rate_delta_mean",
    "early_realized_revision_rate_mean",
    "early_mean_local_loss",
    "early_mean_social_loss",
    "early_action_rate_mean",
    "early_post_social_probability_mean",
    "early_post_local_gt_0p5_rate_mean",
    "early_post_local_gt_0p7_rate_mean",
    "early_post_local_gt_0p9_rate_mean",
    "early_post_local_dwell_0p4_0p6_rate_mean",
    "early_post_social_gt_0p5_rate_mean",
    "early_post_social_gt_0p7_rate_mean",
    "early_post_social_gt_0p9_rate_mean",
    "early_post_social_dwell_0p4_0p6_rate_mean",
    "early_post_social_p10_mean",
    "early_post_social_p50_mean",
    "early_post_social_p90_mean",
    "early_threshold_crossings_0p5_count_sum",
    "early_threshold_crossings_0p7_count_sum",
    "early_threshold_crossings_0p9_count_sum",
    "early_threshold_crossings_0p5_rate_mean",
    "early_threshold_crossings_0p7_rate_mean",
    "early_threshold_crossings_0p9_rate_mean",
    "early_action_flip_rate_mean",
    "bottleneck_flags",
]


PASSTHROUGH_AGGREGATE_DIAGNOSTIC_FIELDS = [
    "policy_action_probability_pre_revision_gt_0p5_rate",
    "policy_action_probability_pre_revision_gt_0p7_rate",
    "policy_action_probability_pre_revision_gt_0p9_rate",
    "policy_action_probability_pre_revision_dwell_0p4_0p6_rate",
    "policy_action_probability_pre_revision_p10",
    "policy_action_probability_pre_revision_p50",
    "policy_action_probability_pre_revision_p90",
    "policy_action_probability_post_local_gt_0p5_rate",
    "policy_action_probability_post_local_gt_0p7_rate",
    "policy_action_probability_post_local_gt_0p9_rate",
    "policy_action_probability_post_local_dwell_0p4_0p6_rate",
    "policy_action_probability_post_local_p10",
    "policy_action_probability_post_local_p50",
    "policy_action_probability_post_local_p90",
    "policy_action_probability_post_social_gt_0p5_rate",
    "policy_action_probability_post_social_gt_0p7_rate",
    "policy_action_probability_post_social_gt_0p9_rate",
    "policy_action_probability_post_social_dwell_0p4_0p6_rate",
    "policy_action_probability_post_social_p10",
    "policy_action_probability_post_social_p50",
    "policy_action_probability_post_social_p90",
    "policy_probability_threshold_crossings_0p5_count",
    "policy_probability_threshold_crossings_0p7_count",
    "policy_probability_threshold_crossings_0p9_count",
    "policy_probability_threshold_crossings_0p5_rate",
    "policy_probability_threshold_crossings_0p7_rate",
    "policy_probability_threshold_crossings_0p9_rate",
    "action_flip_count",
    "action_flip_rate",
]


@dataclass(frozen=True)
class TimeToCeilingDiagnosticResult:
    """Output paths and in-memory rows for a TtC diagnostic run."""

    epoch_path: Path
    run_path: Path
    markdown_path: Path
    epoch_rows: list[dict[str, Any]]
    run_rows: list[dict[str, Any]]


def write_time_to_ceiling_diagnostics(
    *,
    runs_csv: Path,
    output_prefix: Path,
    early_epochs: int = 10,
    min_delta: float = 1e-4,
    decision_gap: float = 0.05,
) -> TimeToCeilingDiagnosticResult:
    """Write derived epoch/run diagnostics from evidence matrix run rows."""

    run_inputs = _read_csv_rows(runs_csv)
    epoch_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run_input in run_inputs:
        run_dir_value = run_input.get("run_dir")
        if not run_dir_value:
            continue
        aggregate_path = Path(str(run_dir_value)) / "aggregate_metrics.csv"
        aggregate_rows = _read_csv_rows(aggregate_path)
        derived_epochs = _derive_epoch_rows(run_input, aggregate_rows)
        epoch_rows.extend(derived_epochs)
        run_rows.append(
            _summarize_run(
                run_input,
                derived_epochs,
                early_epochs=early_epochs,
                min_delta=min_delta,
                decision_gap=decision_gap,
            )
        )

    epoch_path = output_prefix.with_name(f"{output_prefix.name}_epoch_diagnostics.csv")
    run_path = output_prefix.with_name(f"{output_prefix.name}_run_diagnostics.csv")
    markdown_path = output_prefix.with_name(f"{output_prefix.name}_summary.md")
    _write_csv(epoch_path, EPOCH_DIAGNOSTIC_FIELDS, epoch_rows)
    _write_csv(run_path, RUN_DIAGNOSTIC_FIELDS, run_rows)
    _write_markdown(
        markdown_path,
        runs_csv=runs_csv,
        run_rows=run_rows,
        early_epochs=early_epochs,
        min_delta=min_delta,
        decision_gap=decision_gap,
    )
    return TimeToCeilingDiagnosticResult(
        epoch_path=epoch_path,
        run_path=run_path,
        markdown_path=markdown_path,
        epoch_rows=epoch_rows,
        run_rows=run_rows,
    )


def _derive_epoch_rows(
    run_input: dict[str, str],
    aggregate_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    previous_action_rate: float | None = None
    previous_payoff: float | None = None
    for aggregate in aggregate_rows:
        epoch = _optional_int(aggregate.get("epoch"))
        if epoch is None:
            continue
        action_rate = _optional_float(aggregate.get("action_rate"))
        mean_payoff = _optional_float(aggregate.get("mean_payoff"))
        pre_prob = _optional_float(
            aggregate.get("mean_policy_action_probability_pre_revision")
        )
        post_local_prob = _optional_float(
            aggregate.get("mean_policy_action_probability_post_local")
        )
        post_social_prob = _optional_float(
            aggregate.get("mean_policy_action_probability_post_social")
        )
        row: dict[str, Any] = {
            "label": run_input.get("label", ""),
            "case": run_input.get("case", ""),
            "toy": run_input.get("toy", ""),
            "variant": run_input.get("variant", ""),
            "group": run_input.get("group", ""),
            "seed": run_input.get("seed", ""),
            "epoch": epoch,
            "time_to_ceiling": run_input.get("time_to_ceiling", ""),
            "final_within_ceiling": run_input.get("final_within_ceiling", ""),
            "mean_payoff": mean_payoff,
            "action_rate": action_rate,
            "mean_policy_action_probability_pre_revision": pre_prob,
            "mean_policy_action_probability_post_local": post_local_prob,
            "mean_policy_action_probability_post_social": post_social_prob,
            **{
                field: _optional_float(aggregate.get(field))
                for field in PASSTHROUGH_AGGREGATE_DIAGNOSTIC_FIELDS
            },
            "policy_pre_to_local_delta": _subtract(post_local_prob, pre_prob),
            "policy_local_to_social_delta": _subtract(
                post_social_prob,
                post_local_prob,
            ),
            "policy_social_to_action_gap": _subtract(post_social_prob, action_rate),
            "action_rate_delta": _subtract(action_rate, previous_action_rate),
            "mean_payoff_delta": _subtract(mean_payoff, previous_payoff),
            "realized_revision_rate": _optional_float(
                aggregate.get("realized_revision_rate")
            ),
            "mean_local_loss": _optional_float(aggregate.get("mean_local_loss")),
            "mean_social_loss": _optional_float(aggregate.get("mean_social_loss")),
            "domain_effective_advantage_mean": _optional_float(
                aggregate.get("domain_effective_advantage_mean")
            ),
            "domain_effective_advantage_positive_rate": _optional_float(
                aggregate.get("domain_effective_advantage_positive_rate")
            ),
            "domain_basin_action1_advantage_mean": _optional_float(
                aggregate.get("domain_basin_action1_advantage_mean")
            ),
            "domain_basin_action1_advantage_positive_rate": _optional_float(
                aggregate.get("domain_basin_action1_advantage_positive_rate")
            ),
            "domain_basin_training_effective_advantage_mean": _optional_float(
                aggregate.get("domain_basin_training_effective_advantage_mean")
            ),
            "domain_basin_training_effective_advantage_positive_rate": (
                _optional_float(
                    aggregate.get(
                        "domain_basin_training_effective_advantage_positive_rate"
                    )
                )
            ),
            "domain_basin_training_effective_advantage_abs_mean": _optional_float(
                aggregate.get("domain_basin_training_effective_advantage_abs_mean")
            ),
            "domain_basin_credit_positive_rate": _optional_float(
                aggregate.get("domain_basin_credit_positive_rate")
            ),
            "domain_basin_score_delta_mean": _optional_float(
                aggregate.get("domain_basin_score_delta_mean")
            ),
            "domain_resource_fraction": _optional_float(
                aggregate.get("domain_resource_fraction")
            ),
        }
        derived.append(row)
        previous_action_rate = action_rate
        previous_payoff = mean_payoff
    return derived


def _summarize_run(
    run_input: dict[str, str],
    epoch_rows: list[dict[str, Any]],
    *,
    early_epochs: int,
    min_delta: float,
    decision_gap: float,
) -> dict[str, Any]:
    time_to_ceiling = _optional_float(run_input.get("time_to_ceiling"))
    window_rows = [
        row
        for row in epoch_rows
        if int(row["epoch"]) > 0
        and int(row["epoch"]) <= early_epochs
        and (time_to_ceiling is None or int(row["epoch"]) < time_to_ceiling)
    ]
    if not window_rows:
        window_rows = [
            row
            for row in epoch_rows
            if int(row["epoch"]) > 0 and int(row["epoch"]) <= early_epochs
        ]
    first_positive_training = _first_epoch(
        epoch_rows,
        lambda row: _credit_positive(row, min_delta=min_delta),
    )
    first_local_increase = _first_epoch(
        epoch_rows,
        lambda row: _gt(row.get("policy_pre_to_local_delta"), min_delta),
    )
    first_social_nonnegative = _first_epoch(
        epoch_rows,
        lambda row: _ge(row.get("policy_local_to_social_delta"), -min_delta),
    )
    first_action_increase = _first_epoch(
        epoch_rows,
        lambda row: _gt(row.get("action_rate_delta"), min_delta),
    )
    summary = {
        "label": run_input.get("label", ""),
        "case": run_input.get("case", ""),
        "toy": run_input.get("toy", ""),
        "variant": run_input.get("variant", ""),
        "group": run_input.get("group", ""),
        "seed": run_input.get("seed", ""),
        "time_to_ceiling": run_input.get("time_to_ceiling", ""),
        "final_within_ceiling": run_input.get("final_within_ceiling", ""),
        "ever_reached_ceiling": run_input.get("ever_reached_ceiling", ""),
        "window_epoch_count": len(window_rows),
        "first_positive_training_advantage_epoch": first_positive_training,
        "first_local_policy_increase_epoch": first_local_increase,
        "first_social_nonnegative_epoch": first_social_nonnegative,
        "first_action_rate_increase_epoch": first_action_increase,
        "early_effective_advantage_mean": _window_mean(
            window_rows,
            "domain_effective_advantage_mean",
        ),
        "early_effective_advantage_positive_rate": _window_mean(
            window_rows,
            "domain_effective_advantage_positive_rate",
        ),
        "early_basin_action1_advantage_mean": _window_mean(
            window_rows,
            "domain_basin_action1_advantage_mean",
        ),
        "early_basin_action1_advantage_positive_rate": _window_mean(
            window_rows,
            "domain_basin_action1_advantage_positive_rate",
        ),
        "early_training_effective_advantage_mean": _window_mean(
            window_rows,
            "domain_basin_training_effective_advantage_mean",
        ),
        "early_training_effective_advantage_positive_rate": _window_mean(
            window_rows,
            "domain_basin_training_effective_advantage_positive_rate",
        ),
        "early_training_effective_advantage_abs_mean": _window_mean(
            window_rows,
            "domain_basin_training_effective_advantage_abs_mean",
        ),
        "early_policy_pre_to_local_delta_mean": _window_mean(
            window_rows,
            "policy_pre_to_local_delta",
        ),
        "early_policy_local_to_social_delta_mean": _window_mean(
            window_rows,
            "policy_local_to_social_delta",
        ),
        "early_policy_social_to_action_gap_mean": _window_mean(
            window_rows,
            "policy_social_to_action_gap",
        ),
        "early_action_rate_delta_mean": _window_mean(
            window_rows,
            "action_rate_delta",
        ),
        "early_realized_revision_rate_mean": _window_mean(
            window_rows,
            "realized_revision_rate",
        ),
        "early_mean_local_loss": _window_mean(window_rows, "mean_local_loss"),
        "early_mean_social_loss": _window_mean(window_rows, "mean_social_loss"),
        "early_action_rate_mean": _window_mean(window_rows, "action_rate"),
        "early_post_social_probability_mean": _window_mean(
            window_rows,
            "mean_policy_action_probability_post_social",
        ),
        "early_post_local_gt_0p5_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_local_gt_0p5_rate",
        ),
        "early_post_local_gt_0p7_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_local_gt_0p7_rate",
        ),
        "early_post_local_gt_0p9_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_local_gt_0p9_rate",
        ),
        "early_post_local_dwell_0p4_0p6_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_local_dwell_0p4_0p6_rate",
        ),
        "early_post_social_gt_0p5_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_gt_0p5_rate",
        ),
        "early_post_social_gt_0p7_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_gt_0p7_rate",
        ),
        "early_post_social_gt_0p9_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_gt_0p9_rate",
        ),
        "early_post_social_dwell_0p4_0p6_rate_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_dwell_0p4_0p6_rate",
        ),
        "early_post_social_p10_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_p10",
        ),
        "early_post_social_p50_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_p50",
        ),
        "early_post_social_p90_mean": _window_mean(
            window_rows,
            "policy_action_probability_post_social_p90",
        ),
        "early_threshold_crossings_0p5_count_sum": _window_sum(
            window_rows,
            "policy_probability_threshold_crossings_0p5_count",
        ),
        "early_threshold_crossings_0p7_count_sum": _window_sum(
            window_rows,
            "policy_probability_threshold_crossings_0p7_count",
        ),
        "early_threshold_crossings_0p9_count_sum": _window_sum(
            window_rows,
            "policy_probability_threshold_crossings_0p9_count",
        ),
        "early_threshold_crossings_0p5_rate_mean": _window_mean(
            window_rows,
            "policy_probability_threshold_crossings_0p5_rate",
        ),
        "early_threshold_crossings_0p7_rate_mean": _window_mean(
            window_rows,
            "policy_probability_threshold_crossings_0p7_rate",
        ),
        "early_threshold_crossings_0p9_rate_mean": _window_mean(
            window_rows,
            "policy_probability_threshold_crossings_0p9_rate",
        ),
        "early_action_flip_rate_mean": _window_mean(
            window_rows,
            "action_flip_rate",
        ),
    }
    summary["bottleneck_flags"] = ";".join(
        _bottleneck_flags(
            summary,
            time_to_ceiling=time_to_ceiling,
            min_delta=min_delta,
            decision_gap=decision_gap,
        )
    )
    return summary


def _bottleneck_flags(
    summary: dict[str, Any],
    *,
    time_to_ceiling: float | None,
    min_delta: float,
    decision_gap: float,
) -> list[str]:
    flags: list[str] = []
    if str(summary.get("final_within_ceiling", "")).lower() != "true":
        flags.append("no_final_ceiling_hit")

    training_mean = _optional_float(
        summary.get("early_training_effective_advantage_mean")
    )
    training_positive = _optional_float(
        summary.get("early_training_effective_advantage_positive_rate")
    )
    effective_mean = _optional_float(summary.get("early_effective_advantage_mean"))
    effective_positive = _optional_float(
        summary.get("early_effective_advantage_positive_rate")
    )
    credit_mean = training_mean if training_mean is not None else effective_mean
    credit_positive = (
        training_positive if training_positive is not None else effective_positive
    )
    local_delta = _optional_float(summary.get("early_policy_pre_to_local_delta_mean"))
    social_delta = _optional_float(
        summary.get("early_policy_local_to_social_delta_mean")
    )
    decision_lag = _optional_float(
        summary.get("early_policy_social_to_action_gap_mean")
    )
    revision_rate = _optional_float(summary.get("early_realized_revision_rate_mean"))
    post_social_gt_0p5 = _optional_float(
        summary.get("early_post_social_gt_0p5_rate_mean")
    )
    post_social_gt_0p7 = _optional_float(
        summary.get("early_post_social_gt_0p7_rate_mean")
    )
    post_social_dwell = _optional_float(
        summary.get("early_post_social_dwell_0p4_0p6_rate_mean")
    )
    post_social_p10 = _optional_float(summary.get("early_post_social_p10_mean"))
    post_social_p90 = _optional_float(summary.get("early_post_social_p90_mean"))
    crossing_0p5 = _optional_float(
        summary.get("early_threshold_crossings_0p5_rate_mean")
    )
    crossing_0p7 = _optional_float(
        summary.get("early_threshold_crossings_0p7_rate_mean")
    )

    has_credit = (
        (credit_mean is not None and credit_mean > min_delta)
        or (credit_positive is not None and credit_positive >= 0.5)
    )
    if not has_credit:
        flags.append("weak_or_missing_credit_signal")
    if has_credit and (local_delta is None or local_delta <= min_delta):
        flags.append("local_policy_stalled")
    if (
        local_delta is not None
        and local_delta > min_delta
        and social_delta is not None
        and social_delta < -min_delta
    ):
        flags.append("social_dilution")
    if decision_lag is not None and decision_lag > decision_gap:
        flags.append("decision_action_lag")
    if revision_rate is not None and revision_rate < 0.5:
        flags.append("low_revision_rate")
    if post_social_dwell is not None and post_social_dwell >= 0.35:
        flags.append("ambivalence_dwell")
    if (crossing_0p5 is not None and crossing_0p5 >= 0.10) or (
        crossing_0p7 is not None and crossing_0p7 >= 0.10
    ):
        flags.append("threshold_crossing_activity")
    if (
        has_credit
        and local_delta is not None
        and local_delta > min_delta
        and post_social_gt_0p5 is not None
        and post_social_gt_0p5 >= 0.5
        and post_social_gt_0p7 is not None
        and post_social_gt_0p7 < 0.5
    ):
        flags.append("slow_or_partial_commitment")
    if (
        post_social_p10 is not None
        and post_social_p10 <= 0.2
        and post_social_p90 is not None
        and post_social_p90 >= 0.8
    ):
        flags.append("policy_polarization")
    if has_credit and time_to_ceiling is not None and time_to_ceiling > 10:
        flags.append("slow_after_positive_signal")
    if not flags:
        flags.append("no_obvious_early_bottleneck")
    return flags


def _write_markdown(
    path: Path,
    *,
    runs_csv: Path,
    run_rows: list[dict[str, Any]],
    early_epochs: int,
    min_delta: float,
    decision_gap: float,
) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["case"]), str(row["variant"]), str(row["group"]))].append(row)
    lines = [
        "# Time-to-Ceiling Bottleneck Diagnostics",
        "",
        f"Input runs: `{runs_csv}`",
        f"Early window: epochs 1-{early_epochs} before ceiling",
        f"Min delta: `{min_delta}`",
        f"Decision gap threshold: `{decision_gap}`",
        "",
        "## Group Summary",
        "",
        (
            "| Case | Variant | Group | Runs | Final Hits | Mean TtC | "
            "Top Bottlenecks | Local Delta | Social Delta | Decision Gap | "
            "PostSocial >0.7 | Dwell 0.4-0.6 | Cross 0.5 | Flip Rate | "
            "Training Adv | Training Positive Rate |"
        ),
        (
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for (case, variant, group), rows in sorted(grouped.items()):
        final_hits = sum(
            1 for row in rows if str(row.get("final_within_ceiling", "")).lower() == "true"
        )
        flag_counts = Counter(
            flag
            for row in rows
            for flag in str(row.get("bottleneck_flags", "")).split(";")
            if flag
        )
        top_flags = ", ".join(
            f"{flag}:{count}" for flag, count in flag_counts.most_common(3)
        )
        lines.append(
            "| {case} | {variant} | {group} | {runs} | {hits} | {ttc} | "
            "{flags} | {local} | {social} | {decision} | {gt07} | {dwell} | "
            "{cross05} | {flip} | {training} | {positive} |".format(
                case=case,
                variant=variant,
                group=group,
                runs=len(rows),
                hits=final_hits,
                ttc=_format_number(_mean(_optional_float(r.get("time_to_ceiling")) for r in rows)),
                flags=top_flags,
                local=_format_number(
                    _mean(
                        _optional_float(r.get("early_policy_pre_to_local_delta_mean"))
                        for r in rows
                    )
                ),
                social=_format_number(
                    _mean(
                        _optional_float(r.get("early_policy_local_to_social_delta_mean"))
                        for r in rows
                    )
                ),
                decision=_format_number(
                    _mean(
                        _optional_float(r.get("early_policy_social_to_action_gap_mean"))
                        for r in rows
                    )
                ),
                gt07=_format_number(
                    _mean(
                        _optional_float(r.get("early_post_social_gt_0p7_rate_mean"))
                        for r in rows
                    )
                ),
                dwell=_format_number(
                    _mean(
                        _optional_float(
                            r.get("early_post_social_dwell_0p4_0p6_rate_mean")
                        )
                        for r in rows
                    )
                ),
                cross05=_format_number(
                    _mean(
                        _optional_float(
                            r.get("early_threshold_crossings_0p5_rate_mean")
                        )
                        for r in rows
                    )
                ),
                flip=_format_number(
                    _mean(
                        _optional_float(r.get("early_action_flip_rate_mean"))
                        for r in rows
                    )
                ),
                training=_format_number(
                    _mean(
                        _optional_float(r.get("early_training_effective_advantage_mean"))
                        for r in rows
                    )
                ),
                positive=_format_number(
                    _mean(
                        _optional_float(
                            r.get("early_training_effective_advantage_positive_rate")
                        )
                        for r in rows
                    )
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Run Details",
            "",
            (
                "| Case | Variant | Seed | TtC | Final Hit | Bottlenecks | "
                "First Credit | First Local | First Action |"
            ),
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        run_rows,
        key=lambda item: (
            str(item["case"]),
            str(item["variant"]),
            int(str(item["seed"]) or 0),
        ),
    ):
        lines.append(
            "| {case} | {variant} | {seed} | {ttc} | {hit} | {flags} | {credit} | {local} | {action} |".format(
                case=row["case"],
                variant=row["variant"],
                seed=row["seed"],
                ttc=row.get("time_to_ceiling", ""),
                hit=row.get("final_within_ceiling", ""),
                flags=row.get("bottleneck_flags", ""),
                credit=row.get("first_positive_training_advantage_epoch", ""),
                local=row.get("first_local_policy_increase_epoch", ""),
                action=row.get("first_action_rate_increase_epoch", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _gt(value: Any, threshold: float) -> bool:
    number = _optional_float(value)
    return number is not None and number > threshold


def _ge(value: Any, threshold: float) -> bool:
    number = _optional_float(value)
    return number is not None and number >= threshold


def _credit_positive(row: dict[str, Any], *, min_delta: float) -> bool:
    training_mean = _optional_float(
        row.get("domain_basin_training_effective_advantage_mean")
    )
    training_positive = _optional_float(
        row.get("domain_basin_training_effective_advantage_positive_rate")
    )
    effective_mean = _optional_float(row.get("domain_effective_advantage_mean"))
    effective_positive = _optional_float(
        row.get("domain_effective_advantage_positive_rate")
    )
    return (
        (training_mean is not None and training_mean > min_delta)
        or (training_positive is not None and training_positive >= 0.5)
        or (effective_mean is not None and effective_mean > min_delta)
        or (effective_positive is not None and effective_positive >= 0.5)
    )


def _first_epoch(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> int | str:
    for row in rows:
        epoch = int(row["epoch"])
        if epoch > 0 and predicate(row):
            return epoch
    return ""


def _window_mean(rows: list[dict[str, Any]], field: str) -> float | str:
    return _mean(_optional_float(row.get(field)) for row in rows)


def _window_sum(rows: list[dict[str, Any]], field: str) -> float | str:
    numbers = [
        value
        for value in (_optional_float(row.get(field)) for row in rows)
        if value is not None
    ]
    if not numbers:
        return ""
    return float(sum(numbers))


def _mean(values: Iterable[float | None]) -> float | str:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return ""
    return float(sum(numbers) / len(numbers))


def _format_number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"
