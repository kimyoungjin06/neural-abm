#!/usr/bin/env python
"""Run the first Toy 1 ablation matrix from a base config."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from neural_abm.config import load_toy1_config
from neural_abm.toy_classification import run_toy1


@dataclass(frozen=True)
class AblationCase:
    mixer: str
    peer_rule: str
    init_mode: str

    @property
    def slug(self) -> str:
        return f"{self.mixer}_{self.peer_rule}_{self.init_mode}"


FIRST_ABLATION_CASES = [
    AblationCase("none", "none", "same_init"),
    AblationCase("output_average", "output_similarity", "same_init"),
    AblationCase("latent_average", "state_similarity", "same_init"),
    AblationCase("parameter_average", "state_similarity", "same_init"),
    AblationCase("parameter_average", "state_similarity", "independent_init"),
]


SUMMARY_FIELDS = [
    "label",
    "case",
    "seed",
    "coordination_mixer",
    "coordination_peer_rule",
    "model_init_mode",
    "run_dir",
    "domain_final_mean_global_accuracy",
    "domain_final_mean_consensus",
    "final_fragmentation_components",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("experiments/configs/toy1_neural_hk_baseline.yaml"),
        help="Base Toy 1 YAML config.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Ablation label. Defaults to a timestamped first-ablation label.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Seeds to run for each ablation case.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/configs/generated"),
        help="Directory for generated per-run configs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for ablation summaries.",
    )
    return parser.parse_args()


def load_raw_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_case_config(
    base: dict[str, Any],
    case: AblationCase,
    seed: int,
    label: str,
    config_dir: Path,
) -> Path:
    raw = deepcopy(base)
    raw["run"]["name"] = f"{label}_{case.slug}"
    raw["run"]["seed"] = seed
    raw["model"]["agents"]["init_mode"] = case.init_mode
    raw["model"]["coordination"]["mixer"] = case.mixer
    raw["model"]["coordination"]["peer_rule"] = case.peer_rule

    case_dir = config_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{case.slug}_seed{seed:02d}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, label: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Toy 1 Ablation Summary: {label}",
        "",
        "| Case | Seed | Accuracy | Consensus | Fragmentation | Run |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {seed} | {accuracy:.6f} | {consensus:.6f} | "
            "{fragmentation} | `{run_dir}` |".format(
                case=row["case"],
                seed=row["seed"],
                accuracy=row["domain_final_mean_global_accuracy"],
                consensus=row["domain_final_mean_consensus"],
                fragmentation=row["final_fragmentation_components"],
                run_dir=row["run_dir"],
            )
        )
    lines.append("")
    lines.append("This is an automated first ablation summary.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or f"{timestamp}_toy1_first_ablation"
    base = load_raw_yaml(args.base_config)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for case in FIRST_ABLATION_CASES:
            config_path = write_case_config(
                base=base,
                case=case,
                seed=seed,
                label=label,
                config_dir=args.config_dir,
            )
            config = load_toy1_config(config_path)
            result = run_toy1(config=config, config_path=config_path)
            row = {
                "label": label,
                "case": case.slug,
                "seed": seed,
                "coordination_mixer": case.mixer,
                "coordination_peer_rule": case.peer_rule,
                "model_init_mode": case.init_mode,
                "run_dir": str(result.run_dir),
                "domain_final_mean_global_accuracy": result.domain_metrics[
                    "domain_final_mean_global_accuracy"
                ],
                "domain_final_mean_consensus": result.domain_metrics[
                    "domain_final_mean_consensus"
                ],
                "final_fragmentation_components": (
                    result.final_fragmentation_components
                ),
            }
            rows.append(row)
            print(
                f"{case.slug} seed={seed} "
                f"acc={result.domain_metrics['domain_final_mean_global_accuracy']:.6f} "
                f"consensus={result.domain_metrics['domain_final_mean_consensus']:.6f} "
                f"fragmentation={result.final_fragmentation_components}"
            )

    csv_path = args.results_dir / f"{label}_summary.csv"
    md_path = args.results_dir / f"{label}_summary.md"
    write_summary_csv(csv_path, rows)
    write_summary_markdown(md_path, label, rows)
    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
