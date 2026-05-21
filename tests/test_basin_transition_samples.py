from __future__ import annotations

import pytest

from neural_abm.basin_transition_samples import annotate_terminal_outcomes


def test_annotate_terminal_outcomes_adds_future_basin_motion_labels() -> None:
    samples = [
        {
            "run_id": "run",
            "seed": 1,
            "epoch": 1,
            "mean_payoff": 0.2,
            "target_payoff": 1.0,
        },
        {
            "run_id": "run",
            "seed": 1,
            "epoch": 2,
            "mean_payoff": 0.2,
            "target_payoff": 1.0,
        },
        {
            "run_id": "run",
            "seed": 1,
            "epoch": 3,
            "mean_payoff": 0.9,
            "target_payoff": 1.0,
        },
        {
            "run_id": "run",
            "seed": 1,
            "epoch": 4,
            "mean_payoff": 1.0,
            "target_payoff": 1.0,
        },
    ]

    annotated = annotate_terminal_outcomes(
        samples,
        final_mean_payoff=1.0,
        target_payoff=1.0,
        future_horizon=2,
    )

    by_epoch = {int(row["epoch"]): row for row in annotated}
    assert by_epoch[1]["future_basin_horizon"] == 2
    assert by_epoch[1]["future_mean_payoff"] == pytest.approx(0.9)
    assert by_epoch[1]["future_basin_score_delta"] == pytest.approx(0.7)
    assert not bool(by_epoch[1]["future_ceiling_reached"])
    assert by_epoch[2]["future_mean_payoff"] == pytest.approx(1.0)
    assert by_epoch[2]["future_basin_score_delta"] == pytest.approx(0.8)
    assert bool(by_epoch[2]["future_ceiling_reached"])
    assert by_epoch[2]["future_epochs_to_ceiling"] == pytest.approx(2.0)
    assert bool(by_epoch[2]["future_basin_motion_positive"])
