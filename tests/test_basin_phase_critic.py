from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neural_abm.basin_phase_critic import (
    BasinPhaseCriticTrainingConfig,
    BasinReplayWeightScorerTrainingConfig,
    CANDIDATE_CONTEXT_BASIN_PHASE_CRITIC_FEATURES,
    LearnedBasinPhaseCriticBundle,
    LearnedBasinRuntimeDiagnostics,
    basin_phase_critic_examples,
    candidate_action_diagnostics,
    candidate_phase_context_examples,
    learned_basin_credit_signal,
    load_learned_basin_replay_weight_scorer,
    load_basin_phase_critic_manifest,
    run_basin_phase_critic_workflow,
    train_basin_replay_weight_scorer,
    train_evaluate_basin_phase_critic,
)


def synthetic_transition_samples(toy: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        for epoch in range(1, 9):
            mean_payoff = 0.2 if epoch <= 2 else 0.9 if epoch == 3 else 1.0
            action = 1 if epoch >= 3 else 0
            for agent_id in range(2):
                rows.append(
                    {
                        "sample_schema_version": 1,
                        "toy": toy,
                        "run_id": f"{toy}_seed{seed}",
                        "seed": seed,
                        "epoch": epoch,
                        "agent_id": agent_id,
                        "action_observed": action,
                        "policy_action_probability": 0.2 + 0.1 * epoch,
                        "mean_payoff": mean_payoff,
                        "target_payoff": 1.0,
                        "time_to_ceiling": 4,
                        "phase_payoff_alignment": mean_payoff,
                        "phase_action_rate": 0.2 if epoch <= 2 else 1.0,
                        "phase_policy_rate": 0.2 + 0.1 * epoch,
                        "phase_consensus": 1.0 if epoch >= 4 else 0.2,
                        "phase_payoff_stability": 0.5 + 0.05 * epoch,
                        "score_observed": mean_payoff,
                        "basin_action1_advantage": 0.5 if epoch >= 3 else 0.05,
                        "training_effective_advantage": (
                            0.6 if epoch >= 3 else 0.1
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_basin_phase_critic_examples_use_near_term_ceiling_label() -> None:
    config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_horizon=1,
        ceiling_tolerance=0.001,
    )

    examples = basin_phase_critic_examples(
        synthetic_transition_samples("toy2"),
        config=config,
    )

    by_epoch = examples.groupby("epoch")["target_reached_within_horizon"].first()
    assert not bool(by_epoch.loc[1])
    assert not bool(by_epoch.loc[2])
    assert bool(by_epoch.loc[3])
    assert bool(by_epoch.loc[4])


def test_candidate_phase_context_features_recompute_candidate_phase_rates() -> None:
    frame = pd.DataFrame(
        {
            "agent_id": [0, 1],
            "agent_count": [2, 2],
            "action_observed": [0, 1],
            "candidate_action": [1, 0],
            "policy_action_probability": [0.25, 0.75],
            "action_rate": [0.5, 0.5],
            "policy_rate": [0.5, 0.5],
            "phase_action_rate": [0.5, 0.5],
            "phase_policy_rate": [0.5, 0.5],
        }
    )

    context = candidate_phase_context_examples(frame)

    assert context["candidate_action_delta"].to_numpy() == pytest.approx([1.0, -1.0])
    assert context["candidate_policy_delta"].to_numpy() == pytest.approx([0.75, -0.75])
    assert context["candidate_phase_action_rate"].to_numpy() == pytest.approx([1.0, 0.0])
    assert context["candidate_phase_policy_rate"].to_numpy() == pytest.approx(
        [0.875, 0.125]
    )
    assert context["candidate_phase_consensus"].to_numpy() == pytest.approx([1.0, 1.0])


def test_basin_phase_critic_examples_can_use_prototype_direction_labels() -> None:
    config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_mode="prototype_direction",
        label_horizon=1,
        ceiling_tolerance=0.001,
    )

    examples = basin_phase_critic_examples(
        synthetic_transition_samples("toy2"),
        config=config,
    )

    assert set(examples["candidate_action"]) == {0.0, 1.0}
    by_candidate = examples.groupby("candidate_action")[
        "target_reached_within_horizon"
    ].mean()
    assert by_candidate.loc[0.0] == pytest.approx(0.0)
    assert by_candidate.loc[1.0] == pytest.approx(1.0)
    assert examples["prototype_direction_margin"].min() > 0.0


def test_basin_phase_critic_examples_can_use_future_outcome_direction_labels() -> None:
    config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_mode="future_outcome_direction",
        label_horizon=1,
        ceiling_tolerance=0.001,
    )

    examples = basin_phase_critic_examples(
        synthetic_transition_samples("toy2"),
        config=config,
    )

    assert set(examples["candidate_action"]) == {0.0, 1.0}
    by_epoch_candidate = examples.groupby(["epoch", "candidate_action"])[
        "target_reached_within_horizon"
    ].mean()
    assert by_epoch_candidate.loc[(2, 0.0)] == pytest.approx(1.0)
    assert by_epoch_candidate.loc[(2, 1.0)] == pytest.approx(0.0)
    assert by_epoch_candidate.loc[(3, 0.0)] == pytest.approx(0.0)
    assert by_epoch_candidate.loc[(3, 1.0)] == pytest.approx(1.0)
    assert examples["future_direction_margin"].min() > 0.0


def test_train_evaluate_basin_phase_critic_passes_on_separable_samples() -> None:
    config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_horizon=1,
        ceiling_tolerance=0.001,
        max_epochs=80,
        learning_rate=0.05,
        ensemble_size=3,
    )

    result = train_evaluate_basin_phase_critic(
        synthetic_transition_samples("toy2"),
        label="unit",
        case="toy2_case",
        toy="toy2",
        config=config,
    )

    assert result.status == "pass"
    assert result.critic is not None
    assert len(result.ensemble) == 3
    assert result.metrics["eval_auc"] == pytest.approx(1.0)
    assert result.metrics["eval_pairwise_rank_accuracy"] == pytest.approx(1.0)
    assert result.metrics["eval_candidate_uncertainty_mean"] >= 0.0
    assert result.metrics["eval_abstention_rate"] <= 1.0

    examples = basin_phase_critic_examples(
        synthetic_transition_samples("toy2"),
        config=config,
    )
    diagnostics = candidate_action_diagnostics(
        result.ensemble,
        examples,
        config=config,
    )
    assert set(diagnostics) == {
        "candidate_score_action0",
        "candidate_score_action1",
        "candidate_score_std_action0",
        "candidate_score_std_action1",
        "learned_basin_score",
        "learned_basin_action1_advantage",
        "learned_basin_action_margin",
        "learned_basin_uncertainty",
        "learned_basin_abstain",
    }
    assert len(diagnostics["learned_basin_action1_advantage"]) == len(examples)


def test_basin_replay_weight_scorer_trains_and_roundtrips(tmp_path: Path) -> None:
    critic_config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_horizon=1,
        ceiling_tolerance=0.001,
        max_epochs=40,
        learning_rate=0.05,
        ensemble_size=2,
    )
    critic_result = train_evaluate_basin_phase_critic(
        synthetic_transition_samples("toy2"),
        label="unit",
        case="toy2_case",
        toy="toy2",
        config=critic_config,
    )
    assert critic_result.critic is not None

    scorer, metrics = train_basin_replay_weight_scorer(
        synthetic_transition_samples("toy2"),
        critic_bundle=LearnedBasinPhaseCriticBundle(
            main=critic_result.critic,
            ensemble=critic_result.ensemble,
        ),
        config=BasinReplayWeightScorerTrainingConfig(
            train_seeds=(1, 2),
            eval_seeds=(3,),
            max_epochs=40,
            learning_rate=0.05,
            output_floor=0.5,
        ),
    )
    model_path = tmp_path / "replay_weight_scorer.npz"
    scorer.save_npz(model_path)
    loaded = load_learned_basin_replay_weight_scorer(model_path)
    frame = pd.DataFrame({column: [0.0] for column in loaded.feature_columns})

    assert metrics["eval_n"] > 0
    assert metrics["eval_weight_mean"] >= 0.5
    assert loaded.predict_weight(frame)[0] >= 0.5


def test_basin_replay_weight_scorer_can_use_future_motion_target(
    tmp_path: Path,
) -> None:
    critic_config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_horizon=1,
        ceiling_tolerance=0.001,
        max_epochs=40,
        learning_rate=0.05,
        ensemble_size=2,
    )
    critic_result = train_evaluate_basin_phase_critic(
        synthetic_transition_samples("toy2"),
        label="unit",
        case="toy2_case",
        toy="toy2",
        config=critic_config,
    )
    assert critic_result.critic is not None

    scorer, metrics = train_basin_replay_weight_scorer(
        synthetic_transition_samples("toy2"),
        critic_bundle=LearnedBasinPhaseCriticBundle(
            main=critic_result.critic,
            ensemble=critic_result.ensemble,
        ),
        config=BasinReplayWeightScorerTrainingConfig(
            train_seeds=(1, 2),
            eval_seeds=(3,),
            target_mode="future_basin_motion",
            future_horizon=1,
            max_epochs=40,
            learning_rate=0.05,
            output_floor=0.5,
        ),
    )
    model_path = tmp_path / "future_motion_replay_weight_scorer.npz"
    scorer.save_npz(model_path)
    loaded = load_learned_basin_replay_weight_scorer(model_path)
    frame = pd.DataFrame({column: [0.0] for column in loaded.feature_columns})

    assert metrics["target_mode"] == "future_basin_motion"
    assert metrics["target_column"] == "future_basin_score_delta"
    assert metrics["future_horizon"] == 1
    assert metrics["eval_n"] > 0
    assert metrics["eval_target_mean"] >= 0.5
    assert loaded.predict_weight(frame)[0] >= 0.5


def test_basin_replay_weight_scorer_can_use_intervention_pressure_target(
    tmp_path: Path,
) -> None:
    critic_config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_horizon=1,
        ceiling_tolerance=0.001,
        max_epochs=40,
        learning_rate=0.05,
        ensemble_size=2,
    )
    critic_result = train_evaluate_basin_phase_critic(
        synthetic_transition_samples("toy2"),
        label="unit",
        case="toy2_case",
        toy="toy2",
        config=critic_config,
    )
    assert critic_result.critic is not None

    scorer, metrics = train_basin_replay_weight_scorer(
        synthetic_transition_samples("toy2"),
        critic_bundle=LearnedBasinPhaseCriticBundle(
            main=critic_result.critic,
            ensemble=critic_result.ensemble,
        ),
        config=BasinReplayWeightScorerTrainingConfig(
            train_seeds=(1, 2),
            eval_seeds=(3,),
            target_mode="intervention_pressure",
            future_horizon=1,
            max_epochs=40,
            learning_rate=0.05,
            output_floor=0.5,
        ),
    )
    model_path = tmp_path / "intervention_pressure_replay_weight_scorer.npz"
    scorer.save_npz(model_path)
    loaded = load_learned_basin_replay_weight_scorer(model_path)
    frame = pd.DataFrame({column: [0.0] for column in loaded.feature_columns})

    assert metrics["target_mode"] == "intervention_pressure"
    assert metrics["target_column"] == (
        "training_effective_advantage+future_basin_score_delta"
    )
    assert metrics["target_scale_magnitude"] > 0.0
    assert metrics["target_scale_future_basin_motion"] > 0.0
    assert metrics["eval_n"] > 0
    assert loaded.predict_weight(frame)[0] >= 0.5


def test_candidate_context_feature_set_is_trainable_on_synthetic_samples() -> None:
    config = BasinPhaseCriticTrainingConfig(
        train_seeds=(1, 2),
        eval_seeds=(3,),
        label_horizon=1,
        ceiling_tolerance=0.001,
        feature_columns=CANDIDATE_CONTEXT_BASIN_PHASE_CRITIC_FEATURES,
        max_epochs=80,
        learning_rate=0.05,
        ensemble_size=3,
    )

    result = train_evaluate_basin_phase_critic(
        synthetic_transition_samples("toy2"),
        label="unit",
        case="toy2_candidate_context",
        toy="toy2",
        config=config,
    )
    examples = basin_phase_critic_examples(
        synthetic_transition_samples("toy2"),
        config=config,
    )

    assert result.status == "pass"
    assert result.metrics["feature_count"] == len(
        CANDIDATE_CONTEXT_BASIN_PHASE_CRITIC_FEATURES
    )
    for field in [
        "candidate_action_delta",
        "candidate_policy_delta",
        "candidate_phase_action_rate",
        "candidate_phase_policy_rate",
        "candidate_phase_consensus",
    ]:
        assert field in examples.columns


def test_learned_basin_credit_signal_gates_abstaining_agents() -> None:
    runtime = LearnedBasinRuntimeDiagnostics(
        score_action0=pd.Series([0.2, 0.4, 0.6]).to_numpy(),
        score_action1=pd.Series([0.8, 0.3, 0.7]).to_numpy(),
        score_std_action0=pd.Series([0.01, 0.02, 0.03]).to_numpy(),
        score_std_action1=pd.Series([0.02, 0.03, 0.04]).to_numpy(),
        score_observed=pd.Series([0.8, 0.4, 0.7]).to_numpy(),
        action1_advantage=pd.Series([0.6, -0.1, 0.1]).to_numpy(),
        action_margin=pd.Series([0.6, 0.1, 0.1]).to_numpy(),
        uncertainty=pd.Series([0.02, 0.03, 0.04]).to_numpy(),
        abstain=pd.Series([False, True, False]).to_numpy(dtype=bool),
        prototype_action1_advantage_correlation=0.5,
        model_path="critic.npz",
        ensemble_size=3,
        abstention_margin_threshold=0.005,
        uncertainty_threshold=0.05,
    )

    prototype_fallback = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="prototype",
    )
    zero_fallback = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="zero",
    )

    assert prototype_fallback.source == "learned_gated_prototype"
    assert prototype_fallback.learned_credit_used_mask.tolist() == [True, False, True]
    assert prototype_fallback.replay_mask.tolist() == [True, True, True]
    assert prototype_fallback.replay_weight.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert prototype_fallback.replay_selection == "all"
    assert prototype_fallback.action1_advantage == pytest.approx([0.6, 0.3, 0.1])
    assert zero_fallback.source == "learned_gated_zero"
    assert zero_fallback.action1_advantage == pytest.approx([0.6, 0.0, 0.1])

    agreement_selection = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="prototype",
        replay_selection="confident_agreement",
    )
    disagreement_selection = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="prototype",
        replay_selection="confident_disagreement",
    )
    agreement_floor_selection = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="prototype",
        replay_selection="confident_agreement",
        replay_min_selected_rate=2 / 3,
        replay_floor_source="prototype_abs",
    )
    eligible_floor_selection = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="prototype",
        replay_selection="confident_agreement",
        replay_min_selected_rate=1.0,
        replay_floor_source="prototype_abs",
        eligible_mask=pd.Series([True, False, True]).to_numpy(dtype=bool),
    )
    soft_attention = learned_basin_credit_signal(
        runtime,
        prototype_action1_advantage=pd.Series([0.2, 0.3, -0.4]).to_numpy(),
        fallback="prototype",
        replay_selection="confident_agreement",
        replay_mode="soft_attention",
        replay_soft_min_weight=0.25,
        replay_soft_disagreement_weight=0.5,
    )

    assert agreement_selection.replay_mask.tolist() == [True, False, False]
    assert disagreement_selection.replay_mask.tolist() == [False, False, True]
    assert agreement_floor_selection.replay_mask.tolist() == [True, False, True]
    assert eligible_floor_selection.replay_mask.tolist() == [True, False, True]
    assert soft_attention.replay_mask.tolist() == [True, True, True]
    assert soft_attention.replay_weight[0] > soft_attention.replay_weight[2]
    assert soft_attention.replay_weight[1] == pytest.approx(0.25)


def test_basin_phase_critic_workflow_writes_quality_artifacts(
    tmp_path: Path,
) -> None:
    run_rows: list[dict[str, object]] = []
    for toy in ("toy2", "toy4"):
        run_dir = tmp_path / "runs" / toy
        run_dir.mkdir(parents=True)
        synthetic_transition_samples(toy).to_parquet(
            run_dir / "basin_transition_samples.parquet",
            index=False,
        )
        run_rows.append({"toy": toy, "group": "nabm", "run_dir": str(run_dir)})
    runs_csv = tmp_path / "runs.csv"
    with runs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["toy", "group", "run_dir"])
        writer.writeheader()
        writer.writerows(run_rows)

    manifest_path = tmp_path / "critic_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "label": "unit_basin_phase_critic",
                "runs_csv": str(runs_csv),
                "output_dir": str(tmp_path / "critic_results"),
                "train_seeds": [1, 2],
                "eval_seeds": [3],
                "max_epochs": 80,
                "ensemble_size": 3,
                "cases": [
                    {
                        "name": "toy2_case",
                        "toy": "toy2",
                        "ceiling_tolerance": 0.001,
                        "label_horizon": 1,
                    },
                    {
                        "name": "toy4_case",
                        "toy": "toy4",
                        "ceiling_tolerance": 0.001,
                        "label_horizon": 1,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_basin_phase_critic_workflow(manifest_path)

    assert result.summary_csv_path.exists()
    assert result.summary_json_path.exists()
    assert result.markdown_path.exists()
    assert {row["status"] for row in result.rows} == {"pass"}
    for row in result.rows:
        assert Path(str(row["model_path"])).exists()
        assert Path(str(row["predictions_path"])).exists()
        predictions = pd.read_csv(Path(str(row["predictions_path"])))
        for field in [
            "candidate_score_action0",
            "candidate_score_action1",
            "candidate_score_std_action0",
            "candidate_score_std_action1",
            "learned_basin_action1_advantage",
            "learned_basin_uncertainty",
            "learned_basin_abstain",
        ]:
            assert field in predictions.columns


def test_toy24_basin_phase_critic_manifest_contract() -> None:
    manifest = load_basin_phase_critic_manifest(
        Path("experiments/evidence/toy24_basin_phase_critic_quality_quick.yaml")
    )

    assert manifest.label == "toy24_basin_phase_critic_quality_quick"
    assert manifest.train_seeds == (1, 2)
    assert manifest.eval_seeds == (3,)
    assert manifest.ensemble_size == 5
    assert manifest.abstention_margin_threshold == pytest.approx(0.005)
    assert manifest.uncertainty_threshold == pytest.approx(0.05)
    assert manifest.feature_columns == (
        "candidate_action",
        "policy_action_probability",
        "phase_payoff_alignment",
        "phase_action_rate",
        "phase_policy_rate",
        "phase_consensus",
        "phase_payoff_stability",
    )
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}


def test_toy24_basin_phase_critic_candidate_context_manifest_contract() -> None:
    manifest = load_basin_phase_critic_manifest(
        Path("experiments/evidence/toy24_basin_phase_critic_candidate_context_quick.yaml")
    )

    assert manifest.label == "toy24_basin_phase_critic_candidate_context_quick"
    assert manifest.feature_columns == CANDIDATE_CONTEXT_BASIN_PHASE_CRITIC_FEATURES


def test_toy24_basin_phase_critic_future_outcome_manifest_contract() -> None:
    manifest = load_basin_phase_critic_manifest(
        Path(
            "experiments/evidence/"
            "toy24_basin_phase_critic_future_outcome_direction_quick.yaml"
        )
    )

    assert manifest.label == "toy24_basin_phase_critic_future_outcome_direction_quick"
    assert manifest.feature_columns == CANDIDATE_CONTEXT_BASIN_PHASE_CRITIC_FEATURES
    assert {case.label_mode for case in manifest.cases} == {
        "future_outcome_direction"
    }
    assert {case.toy for case in manifest.cases} == {"toy2", "toy4"}
