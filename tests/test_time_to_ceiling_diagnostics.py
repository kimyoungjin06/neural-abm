from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neural_abm.time_to_ceiling_diagnostics import (
    write_time_to_ceiling_diagnostics,
)


RUN_FIELDS = [
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "run_dir",
    "final_within_ceiling",
    "ever_reached_ceiling",
    "time_to_ceiling",
]


AGGREGATE_FIELDS = [
    "epoch",
    "mean_payoff",
    "action_rate",
    "mean_policy_action_probability_pre_revision",
    "mean_policy_action_probability_post_local",
    "mean_policy_action_probability_post_social",
    "policy_action_probability_post_local_gt_0p7_rate",
    "policy_action_probability_post_local_dwell_0p4_0p6_rate",
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
    "action_flip_rate",
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_row(
    epoch: int,
    *,
    action_rate: float,
    pre: float,
    post_local: float,
    post_social: float,
    training_advantage: float,
    realized_revision_rate: float = 1.0,
    mean_payoff: float = 0.5,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "mean_payoff": mean_payoff,
        "action_rate": action_rate,
        "mean_policy_action_probability_pre_revision": pre,
        "mean_policy_action_probability_post_local": post_local,
        "mean_policy_action_probability_post_social": post_social,
        "policy_action_probability_post_local_gt_0p7_rate": 0.25,
        "policy_action_probability_post_local_dwell_0p4_0p6_rate": 0.5,
        "policy_action_probability_post_social_gt_0p5_rate": 0.75,
        "policy_action_probability_post_social_gt_0p7_rate": 0.25,
        "policy_action_probability_post_social_gt_0p9_rate": 0.0,
        "policy_action_probability_post_social_dwell_0p4_0p6_rate": 0.5,
        "policy_action_probability_post_social_p10": 0.2,
        "policy_action_probability_post_social_p50": post_social,
        "policy_action_probability_post_social_p90": 0.8,
        "policy_probability_threshold_crossings_0p5_count": 1,
        "policy_probability_threshold_crossings_0p7_count": 0,
        "policy_probability_threshold_crossings_0p9_count": 0,
        "policy_probability_threshold_crossings_0p5_rate": 0.25,
        "policy_probability_threshold_crossings_0p7_rate": 0.0,
        "policy_probability_threshold_crossings_0p9_rate": 0.0,
        "action_flip_rate": 0.25,
        "realized_revision_rate": realized_revision_rate,
        "mean_local_loss": 0.2,
        "mean_social_loss": 0.1,
        "domain_effective_advantage_mean": training_advantage,
        "domain_effective_advantage_positive_rate": 1.0,
        "domain_basin_action1_advantage_mean": training_advantage,
        "domain_basin_action1_advantage_positive_rate": 1.0,
        "domain_basin_training_effective_advantage_mean": training_advantage,
        "domain_basin_training_effective_advantage_positive_rate": 1.0,
        "domain_basin_training_effective_advantage_abs_mean": abs(training_advantage),
        "domain_basin_credit_positive_rate": 1.0,
        "domain_basin_score_delta_mean": 0.01,
        "domain_resource_fraction": 0.6,
    }


def write_run(
    base: Path,
    *,
    variant: str,
    seed: int,
    final_hit: bool,
    time_to_ceiling: int | str,
    aggregate_rows: list[dict[str, object]],
) -> dict[str, object]:
    run_dir = base / f"{variant}_seed{seed}"
    write_csv(run_dir / "aggregate_metrics.csv", AGGREGATE_FIELDS, aggregate_rows)
    return {
        "label": "unit",
        "case": "toy2_basin_credit",
        "toy": "toy2",
        "variant": variant,
        "group": "nabm",
        "seed": seed,
        "run_dir": str(run_dir),
        "final_within_ceiling": final_hit,
        "ever_reached_ceiling": time_to_ceiling != "",
        "time_to_ceiling": time_to_ceiling,
    }


def test_time_to_ceiling_diagnostics_writes_outputs_and_epoch_deltas(
    tmp_path: Path,
) -> None:
    run_row = write_run(
        tmp_path,
        variant="objective_basin",
        seed=1,
        final_hit=True,
        time_to_ceiling=4,
        aggregate_rows=[
            aggregate_row(
                1,
                action_rate=0.20,
                pre=0.30,
                post_local=0.45,
                post_social=0.40,
                training_advantage=0.2,
                mean_payoff=1.0,
            ),
            aggregate_row(
                2,
                action_rate=0.25,
                pre=0.40,
                post_local=0.50,
                post_social=0.48,
                training_advantage=0.2,
                mean_payoff=1.1,
            ),
        ],
    )
    runs_csv = tmp_path / "runs.csv"
    write_csv(runs_csv, RUN_FIELDS, [run_row])

    result = write_time_to_ceiling_diagnostics(
        runs_csv=runs_csv,
        output_prefix=tmp_path / "diagnostics",
        early_epochs=3,
    )

    assert result.epoch_path.exists()
    assert result.run_path.exists()
    assert result.markdown_path.exists()
    first_epoch = result.epoch_rows[0]
    assert first_epoch["policy_pre_to_local_delta"] == pytest.approx(0.15)
    assert first_epoch["policy_local_to_social_delta"] == pytest.approx(-0.05)
    assert first_epoch["policy_social_to_action_gap"] == pytest.approx(0.20)
    assert result.epoch_rows[1]["action_rate_delta"] == pytest.approx(0.05)
    assert result.epoch_rows[1]["mean_payoff_delta"] == pytest.approx(0.1)
    run_row = result.run_rows[0]
    assert run_row["early_post_social_gt_0p7_rate_mean"] == pytest.approx(0.25)
    assert run_row["early_post_social_dwell_0p4_0p6_rate_mean"] == pytest.approx(0.5)
    assert run_row["early_threshold_crossings_0p5_count_sum"] == pytest.approx(2.0)
    assert run_row["early_action_flip_rate_mean"] == pytest.approx(0.25)


def test_time_to_ceiling_diagnostics_flags_separate_bottlenecks(
    tmp_path: Path,
) -> None:
    social_dilution = write_run(
        tmp_path,
        variant="social_dilution",
        seed=1,
        final_hit=False,
        time_to_ceiling="",
        aggregate_rows=[
            aggregate_row(
                1,
                action_rate=0.30,
                pre=0.30,
                post_local=0.50,
                post_social=0.40,
                training_advantage=0.2,
            )
        ],
    )
    local_stall = write_run(
        tmp_path,
        variant="local_stall",
        seed=2,
        final_hit=False,
        time_to_ceiling="",
        aggregate_rows=[
            aggregate_row(
                1,
                action_rate=0.30,
                pre=0.50,
                post_local=0.50,
                post_social=0.50,
                training_advantage=0.2,
                realized_revision_rate=0.25,
            )
        ],
    )
    runs_csv = tmp_path / "runs.csv"
    write_csv(runs_csv, RUN_FIELDS, [social_dilution, local_stall])

    result = write_time_to_ceiling_diagnostics(
        runs_csv=runs_csv,
        output_prefix=tmp_path / "diagnostics",
        early_epochs=2,
        decision_gap=0.05,
    )

    by_variant = {row["variant"]: row for row in result.run_rows}
    social_flags = by_variant["social_dilution"]["bottleneck_flags"].split(";")
    stall_flags = by_variant["local_stall"]["bottleneck_flags"].split(";")
    assert "social_dilution" in social_flags
    assert "decision_action_lag" in social_flags
    assert "local_policy_stalled" in stall_flags
    assert "low_revision_rate" in stall_flags
