#!/usr/bin/env python
"""Train frozen basin replay-weight scorers from transition samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from neural_abm.basin_phase_critic import (
    BasinReplayWeightScorerTrainingConfig,
    artifact_paths_from_runs_csv,
    load_basin_transition_sample_artifacts,
    load_learned_basin_phase_critic_bundle,
    train_basin_replay_weight_scorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path(
            "experiments/results/nabm_effect_matrix/"
            "toy24_basin_transition_dataset_quick_runs.csv"
        ),
        help="Run CSV containing basin_transition_samples.parquet producers.",
    )
    parser.add_argument(
        "--label",
        default="toy24_basin_replay_weight_scorer_quick",
        help="Output label prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/basin_critic"),
        help="Directory where scorer artifacts are written.",
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[3])
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--target-mode",
        choices=("magnitude", "future_basin_motion", "intervention_pressure"),
        default="magnitude",
        help=(
            "Replay-weight target supervision: target magnitude, positive "
            "future basin payoff motion, or their intervention-pressure max."
        ),
    )
    parser.add_argument(
        "--target-column",
        default="training_effective_advantage",
        help="Column used when --target-mode=magnitude.",
    )
    parser.add_argument("--target-quantile", type=float, default=0.90)
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=5,
        help="Forward epoch window for --target-mode=future_basin_motion.",
    )
    parser.add_argument("--output-floor", type=float, default=0.50)
    parser.add_argument(
        "--toy2-critic-model-path",
        type=Path,
        default=Path(
            "experiments/results/basin_critic/"
            "toy24_basin_phase_critic_candidate_context_quick/"
            "toy24_basin_phase_critic_candidate_context_quick_"
            "toy2_basin_phase_critic_candidate_context_model.npz"
        ),
    )
    parser.add_argument(
        "--toy4-critic-model-path",
        type=Path,
        default=Path(
            "experiments/results/basin_critic/"
            "toy24_basin_phase_critic_candidate_context_quick/"
            "toy24_basin_phase_critic_candidate_context_quick_"
            "toy4_basin_phase_critic_candidate_context_model.npz"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    critic_paths = {
        "toy2": args.toy2_critic_model_path,
        "toy4": args.toy4_critic_model_path,
    }
    for toy, critic_path in critic_paths.items():
        artifact_paths = artifact_paths_from_runs_csv(
            args.runs_csv,
            toy=toy,
            group="nabm",
        )
        samples = load_basin_transition_sample_artifacts(artifact_paths)
        critic_bundle = load_learned_basin_phase_critic_bundle(critic_path)
        scorer, metrics = train_basin_replay_weight_scorer(
            samples,
            critic_bundle=critic_bundle,
            config=BasinReplayWeightScorerTrainingConfig(
                train_seeds=tuple(args.train_seeds),
                eval_seeds=tuple(args.eval_seeds),
                max_epochs=int(args.max_epochs),
                learning_rate=float(args.learning_rate),
                target_mode=args.target_mode,
                target_column=str(args.target_column),
                target_quantile=float(args.target_quantile),
                future_horizon=int(args.future_horizon),
                output_floor=float(args.output_floor),
            ),
        )
        case = f"{toy}_basin_replay_weight_scorer"
        model_path = output_dir / f"{args.label}_{case}_model.npz"
        scorer.save_npz(model_path)
        row = {
            "label": args.label,
            "case": case,
            "toy": toy,
            "critic_model_path": str(critic_path),
            "model_path": str(model_path),
            **metrics,
        }
        rows.append(row)
        (output_dir / f"{args.label}_{case}_summary.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "{case}: eval_mse={mse:.6g}, eval_weight_mean={weight:.6g}, "
            "model={model}".format(
                case=case,
                mse=float(row["eval_mse"]),
                weight=float(row["eval_weight_mean"]),
                model=model_path,
            )
        )
    summary_csv = args.output_dir / f"{args.label}_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown = args.output_dir / f"{args.label}_summary.md"
    markdown.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote scorer summary CSV: {summary_csv}")
    print(f"Wrote scorer report: {markdown}")


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Basin Replay Weight Scorer",
        "",
        "| Case | Toy | Target | Horizon | Train N | Eval N | Eval MSE | Eval Corr | Eval Weight | Eval Target | Model |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {toy} | {target} | {horizon} | {train_n} | {eval_n} | "
            "{mse:.6g} | {corr} | {weight:.6g} | {target_mean:.6g} | "
            "`{model}` |".format(
                case=row["case"],
                toy=row["toy"],
                target=row["target_mode"],
                horizon=row["future_horizon"],
                train_n=row["train_n"],
                eval_n=row["eval_n"],
                mse=float(row["eval_mse"]),
                corr=_format_optional(row["eval_target_correlation"]),
                weight=float(row["eval_weight_mean"]),
                target_mean=float(row["eval_target_mean"]),
                model=row["model_path"],
            )
        )
    return "\n".join(lines) + "\n"


def _format_optional(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.6g}"


if __name__ == "__main__":
    main()
