from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from neural_abm.basin_learned_diagnostic_summary import (
    summarize_basin_learned_diagnostics,
)
from neural_abm.evidence_matrix import load_manifest


def write_run_artifacts(
    run_dir: Path,
    *,
    prototype: list[float],
    learned: list[float],
    abstain: list[bool],
) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [1 for _ in prototype],
            "agent_id": list(range(len(prototype))),
            "domain_basin_action1_advantage": prototype,
            "domain_basin_learned_action1_advantage": learned,
            "domain_basin_learned_uncertainty": [0.01 for _ in prototype],
            "domain_basin_learned_abstain": abstain,
        }
    ).to_csv(run_dir / "micro_state.csv", index=False)
    pd.DataFrame(
        {
            "epoch": [1],
            "domain_basin_training_replay_selection": ["confident_agreement"],
            "domain_basin_training_replay_min_selected_rate": [0.25],
            "domain_basin_training_replay_selected_rate": [0.25],
            "domain_basin_training_replay_weight_mean": [0.6],
            "domain_basin_training_replay_weight_positive_rate": [0.75],
            "domain_basin_training_learned_credit_rate": [0.5],
            "domain_basin_learned_abstention_rate": [sum(abstain) / len(abstain)],
            "domain_basin_learned_prototype_advantage_correlation": [0.5],
        }
    ).to_csv(run_dir / "aggregate_metrics.csv", index=False)


def test_summarize_basin_learned_diagnostics_writes_run_and_group_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    write_run_artifacts(
        run_dir,
        prototype=[1.0, -1.0, 1.0, -1.0],
        learned=[0.5, -0.5, -0.25, 0.25],
        abstain=[False, False, True, True],
    )
    runs_csv = tmp_path / "runs.csv"
    with runs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "case",
                "toy",
                "variant",
                "group",
                "seed",
                "run_dir",
                "final_within_ceiling",
                "time_to_ceiling",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": "unit",
                "case": "toy2_case",
                "toy": "toy2",
                "variant": "readonly",
                "group": "nabm",
                "seed": 1,
                "run_dir": str(run_dir),
                "final_within_ceiling": True,
                "time_to_ceiling": 3,
            }
        )

    result = summarize_basin_learned_diagnostics(
        runs_csv,
        output_dir=tmp_path / "out",
    )

    assert result.run_summary_path.exists()
    assert result.group_summary_path.exists()
    assert result.markdown_path.exists()
    [run_row] = result.run_rows
    assert run_row["diagnostic_status"] == "complete"
    assert run_row["learned_abstention_rate"] == pytest.approx(0.5)
    assert run_row["prototype_learned_sign_agreement_rate"] == pytest.approx(0.5)
    assert run_row["non_abstain_sign_agreement_rate"] == pytest.approx(1.0)
    assert run_row["aggregate_training_replay_selected_rate_mean"] == pytest.approx(
        0.25
    )
    assert run_row[
        "aggregate_training_replay_min_selected_rate_mean"
    ] == pytest.approx(0.25)
    assert run_row["aggregate_training_learned_credit_rate_mean"] == pytest.approx(0.5)
    assert run_row["aggregate_training_replay_weight_mean"] == pytest.approx(0.6)
    assert run_row[
        "aggregate_training_replay_weight_positive_rate_mean"
    ] == pytest.approx(0.75)
    assert run_row["final_aggregate_training_replay_selection"] == (
        "confident_agreement"
    )
    assert run_row["final_aggregate_training_replay_weight_mean"] == pytest.approx(
        0.6
    )
    [group_row] = result.group_rows
    assert group_row["complete_run_count"] == 1
    assert group_row["final_ceiling_hits"] == 1
    assert group_row["prototype_learned_sign_agreement_rate_mean"] == pytest.approx(
        0.5
    )
    assert group_row["aggregate_training_replay_selected_rate_mean"] == pytest.approx(
        0.25
    )
    assert group_row[
        "aggregate_training_replay_min_selected_rate_mean"
    ] == pytest.approx(0.25)
    assert group_row["aggregate_training_replay_weight_mean"] == pytest.approx(0.6)


def test_toy24_basin_learned_diagnostic_manifest_contract() -> None:
    manifest = load_manifest(
        Path("experiments/evidence/toy24_basin_learned_diagnostic_quick.yaml")
    )

    assert manifest.label == "toy24_basin_learned_diagnostic_quick"
    assert manifest.seeds == (1, 2, 3)
    assert manifest.epochs == 50
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
    for case in manifest.cases:
        nabm = [variant for variant in case.variants if variant.group == "nabm"]
        assert len(nabm) == 1
        updates = nabm[0].updates
        assert updates["model.policy.domain.basin_credit.learned_diagnostic_enabled"]
        assert "learned_diagnostic_model_path" in " ".join(updates)
