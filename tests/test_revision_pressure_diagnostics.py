from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neural_abm.revision_pressure_diagnostics import (
    write_revision_pressure_diagnostics,
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
    "action_rate",
    "mean_policy_action_probability_post_local",
    "mean_policy_action_probability_post_social",
    "policy_action_probability_post_social_gt_0p7_rate",
    "domain_basin_training_effective_advantage_mean",
    "domain_basin_training_effective_advantage_positive_rate",
    "domain_effective_advantage_mean",
    "domain_effective_advantage_positive_rate",
    "realized_revision_rate",
    "precommitment_rate",
    "precommitment_mean_evidence",
    "commitment_rate",
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
    post_local: float,
    post_social: float,
    gt_0p7: float,
    training_advantage: float | str,
    training_positive_rate: float | str = 1.0,
    realized_revision_rate: float = 1.0,
    precommitment_rate: float = 0.0,
    commitment_rate: float = 0.0,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "action_rate": action_rate,
        "mean_policy_action_probability_post_local": post_local,
        "mean_policy_action_probability_post_social": post_social,
        "policy_action_probability_post_social_gt_0p7_rate": gt_0p7,
        "domain_basin_training_effective_advantage_mean": training_advantage,
        "domain_basin_training_effective_advantage_positive_rate": (
            training_positive_rate
        ),
        "domain_effective_advantage_mean": "",
        "domain_effective_advantage_positive_rate": "",
        "realized_revision_rate": realized_revision_rate,
        "precommitment_rate": precommitment_rate,
        "precommitment_mean_evidence": precommitment_rate,
        "commitment_rate": commitment_rate,
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


def test_revision_pressure_diagnostics_tracks_pressure_before_policy_and_action(
    tmp_path: Path,
) -> None:
    run_row = write_run(
        tmp_path,
        variant="pressure_before_policy",
        seed=1,
        final_hit=True,
        time_to_ceiling=6,
        aggregate_rows=[
            aggregate_row(
                1,
                action_rate=0.40,
                post_local=0.45,
                post_social=0.50,
                gt_0p7=0.0,
                training_advantage=0.20,
            ),
            aggregate_row(
                2,
                action_rate=0.50,
                post_local=0.55,
                post_social=0.60,
                gt_0p7=0.25,
                training_advantage=0.20,
            ),
            aggregate_row(
                3,
                action_rate=0.70,
                post_local=0.65,
                post_social=0.72,
                gt_0p7=0.50,
                training_advantage=0.20,
            ),
            aggregate_row(
                4,
                action_rate=0.92,
                post_local=0.80,
                post_social=0.85,
                gt_0p7=0.75,
                training_advantage=0.20,
            ),
        ],
    )
    runs_csv = tmp_path / "runs.csv"
    write_csv(runs_csv, RUN_FIELDS, [run_row])

    result = write_revision_pressure_diagnostics(
        runs_csv=runs_csv,
        output_prefix=tmp_path / "revision_pressure",
        early_epochs=5,
        pressure_threshold=0.05,
        policy_threshold=0.7,
        policy_ready_rate_threshold=0.5,
        action_threshold=0.9,
    )

    assert result.epoch_path.exists()
    assert result.run_path.exists()
    assert result.markdown_path.exists()
    first_epoch = result.epoch_rows[0]
    assert first_epoch["revision_pressure_proxy"] == pytest.approx(0.20)
    assert first_epoch["social_revision_pressure_proxy"] == pytest.approx(0.05)
    assert first_epoch["pressure_action_gap"] == pytest.approx(0.10)
    summary = result.run_rows[0]
    assert summary["first_revision_pressure_epoch"] == 1
    assert summary["first_policy_readiness_epoch"] == 3
    assert summary["first_action_response_epoch"] == 4
    assert summary["pressure_to_policy_lag"] == 2
    assert summary["pressure_to_action_lag"] == 3
    flags = summary["revision_framing_flags"].split(";")
    assert "revision_pressure_precedes_policy" in flags
    assert "revision_pressure_precedes_action" in flags
    assert "pressure_action_translation_lag" in flags


def test_revision_pressure_diagnostics_falls_back_to_effective_advantage(
    tmp_path: Path,
) -> None:
    run_row = write_run(
        tmp_path,
        variant="effective_fallback",
        seed=1,
        final_hit=False,
        time_to_ceiling="",
        aggregate_rows=[
            {
                **aggregate_row(
                    1,
                    action_rate=0.30,
                    post_local=0.40,
                    post_social=0.45,
                    gt_0p7=0.0,
                    training_advantage="",
                    training_positive_rate="",
                    realized_revision_rate=0.25,
                ),
                "domain_effective_advantage_mean": 0.30,
                "domain_effective_advantage_positive_rate": 0.80,
            }
        ],
    )
    runs_csv = tmp_path / "runs.csv"
    write_csv(runs_csv, RUN_FIELDS, [run_row])

    result = write_revision_pressure_diagnostics(
        runs_csv=runs_csv,
        output_prefix=tmp_path / "revision_pressure",
        pressure_threshold=0.05,
    )

    summary = result.run_rows[0]
    assert result.epoch_rows[0]["domain_revision_advantage_mean"] == pytest.approx(
        0.30
    )
    assert result.epoch_rows[0]["revision_pressure_proxy"] == pytest.approx(0.24)
    flags = summary["revision_framing_flags"].split(";")
    assert "low_realized_revision_rate" in flags
    assert "pressure_not_sufficient_for_ceiling" in flags
