#!/usr/bin/env python
"""Summarize Toy4/Toy5 stage timings as steady-state step costs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable


TOP_LEVEL_STEADY_STAGES = ("sample_revision_mask", "hooked_step_total")
VIRTUAL_STEADY_STAGE = "steady_state_total"
VIRTUAL_STAGES = (VIRTUAL_STEADY_STAGE,)
EXCLUDED_STAGES = frozenset(
    {
        "make_run_dir",
        "write_metadata",
        "initial_state",
        "writer_setup",
        "initial_step_result",
        "initial_aggregate_row",
        "write_initial_aggregate",
        "aggregate_row",
        "write_aggregate",
        "write_micro",
    }
)
FOCUS_STAGES = (
    "policy_readout",
    "local_loss_update",
    "local_optimizer_update",
    "local_adam_update",
    "social_step",
    "social_distillation",
    "social_loss_update",
    "social_mix",
    "social_optimizer_update",
    "social_adam_update",
    "select_peers",
)

SUMMARY_FIELDS = [
    "toy",
    "device",
    "agent_count",
    "training_backend",
    "stage",
    "repeats",
    "epochs_per_repeat",
    "seconds_per_run_mean",
    "seconds_per_run_std",
    "seconds_per_epoch_mean",
    "seconds_per_epoch_std",
    "share_of_steady_state_pct",
    "share_of_hooked_step_pct",
]

COMPARISON_FIELDS = [
    "toy",
    "device",
    "agent_count",
    "stage",
    "batched_seconds_per_epoch",
    "tensor_batched_seconds_per_epoch",
    "speedup",
    "tensor_minus_batched_seconds_per_epoch",
    "tensor_share_of_steady_state_pct",
    "tensor_share_of_hooked_step_pct",
]


@dataclass(frozen=True)
class RunKey:
    toy: str
    device: str
    agent_count: int
    training_backend: str
    repeat_index: int
    run_id: str


@dataclass(frozen=True)
class StageKey:
    toy: str
    device: str
    agent_count: int
    training_backend: str
    stage: str


@dataclass(frozen=True)
class ComparisonKey:
    toy: str
    device: str
    agent_count: int
    stage: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-timings",
        type=Path,
        required=True,
        help="Detailed *_stage_timings.csv produced by benchmark_toy_gpu_core.py.",
    )
    parser.add_argument(
        "--stage-summary-output",
        type=Path,
        required=True,
        help="CSV with per-backend steady-state stage summaries.",
    )
    parser.add_argument(
        "--backend-comparison-output",
        type=Path,
        required=True,
        help="CSV comparing batched and tensor_batched per stage.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional markdown summary of speedups and top tensor bottlenecks.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=6,
        help="Number of focus stages to show per tensor_batched case in markdown.",
    )
    return parser.parse_args()


def read_stage_timings(
    path: Path,
) -> tuple[dict[RunKey, dict[str, float]], dict[RunKey, int], set[str]]:
    totals: dict[RunKey, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    epochs: dict[RunKey, set[int]] = defaultdict(set)
    stages: set[str] = set()

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if epoch <= 0:
                continue
            stage = row["stage"]
            key = RunKey(
                toy=row["toy"],
                device=row["device"],
                agent_count=int(row["agent_count"]),
                training_backend=row["training_backend"],
                repeat_index=int(row["repeat_index"]),
                run_id=row["run_id"],
            )
            stages.add(stage)
            totals[key][stage] += float(row["seconds"])
            if stage == "hooked_step_total":
                epochs[key].add(epoch)

    epoch_counts = {key: len(value) for key, value in epochs.items()}
    return totals, epoch_counts, stages


def mean(values: Iterable[float]) -> float:
    numeric = list(values)
    if not numeric:
        return 0.0
    return fmean(numeric)


def sample_std(values: Iterable[float]) -> float:
    numeric = list(values)
    if len(numeric) < 2:
        return 0.0
    return stdev(numeric)


def build_stage_summary(
    totals: dict[RunKey, dict[str, float]],
    epoch_counts: dict[RunKey, int],
    stages: set[str],
) -> list[dict[str, object]]:
    stage_universe = sorted((stages - EXCLUDED_STAGES) | set(VIRTUAL_STAGES))
    run_rows: dict[StageKey, list[dict[str, float]]] = defaultdict(list)

    for run_key, run_totals in totals.items():
        epochs = epoch_counts.get(run_key, 0)
        if epochs == 0:
            continue
        steady_total = sum(run_totals.get(stage, 0.0) for stage in TOP_LEVEL_STEADY_STAGES)
        hook_total = run_totals.get("hooked_step_total", 0.0)
        run_totals = dict(run_totals)
        run_totals[VIRTUAL_STEADY_STAGE] = steady_total

        for stage in stage_universe:
            total = run_totals.get(stage, 0.0)
            key = StageKey(
                toy=run_key.toy,
                device=run_key.device,
                agent_count=run_key.agent_count,
                training_backend=run_key.training_backend,
                stage=stage,
            )
            run_rows[key].append(
                {
                    "epochs": float(epochs),
                    "seconds_per_run": total,
                    "seconds_per_epoch": total / epochs,
                    "share_of_steady_state_pct": (
                        100.0 * total / steady_total if steady_total else 0.0
                    ),
                    "share_of_hooked_step_pct": (
                        100.0 * total / hook_total if hook_total else 0.0
                    ),
                }
            )

    summaries: list[dict[str, object]] = []
    for key, records in sorted(
        run_rows.items(),
        key=lambda item: (
            item[0].toy,
            item[0].device,
            item[0].agent_count,
            item[0].training_backend,
            item[0].stage,
        ),
    ):
        summaries.append(
            {
                "toy": key.toy,
                "device": key.device,
                "agent_count": key.agent_count,
                "training_backend": key.training_backend,
                "stage": key.stage,
                "repeats": len(records),
                "epochs_per_repeat": mean(record["epochs"] for record in records),
                "seconds_per_run_mean": mean(
                    record["seconds_per_run"] for record in records
                ),
                "seconds_per_run_std": sample_std(
                    record["seconds_per_run"] for record in records
                ),
                "seconds_per_epoch_mean": mean(
                    record["seconds_per_epoch"] for record in records
                ),
                "seconds_per_epoch_std": sample_std(
                    record["seconds_per_epoch"] for record in records
                ),
                "share_of_steady_state_pct": mean(
                    record["share_of_steady_state_pct"] for record in records
                ),
                "share_of_hooked_step_pct": mean(
                    record["share_of_hooked_step_pct"] for record in records
                ),
            }
        )
    return summaries


def build_backend_comparison(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key: dict[ComparisonKey, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summaries:
        key = ComparisonKey(
            toy=str(row["toy"]),
            device=str(row["device"]),
            agent_count=int(row["agent_count"]),
            stage=str(row["stage"]),
        )
        by_key[key][str(row["training_backend"])] = row

    comparisons: list[dict[str, object]] = []
    for key, backends in sorted(
        by_key.items(),
        key=lambda item: (item[0].toy, item[0].device, item[0].agent_count, item[0].stage),
    ):
        if "batched" not in backends or "tensor_batched" not in backends:
            continue
        batched = float(backends["batched"]["seconds_per_epoch_mean"])
        tensor = float(backends["tensor_batched"]["seconds_per_epoch_mean"])
        comparisons.append(
            {
                "toy": key.toy,
                "device": key.device,
                "agent_count": key.agent_count,
                "stage": key.stage,
                "batched_seconds_per_epoch": batched,
                "tensor_batched_seconds_per_epoch": tensor,
                "speedup": speedup(batched, tensor),
                "tensor_minus_batched_seconds_per_epoch": tensor - batched,
                "tensor_share_of_steady_state_pct": backends["tensor_batched"][
                    "share_of_steady_state_pct"
                ],
                "tensor_share_of_hooked_step_pct": backends["tensor_batched"][
                    "share_of_hooked_step_pct"
                ],
            }
        )
    return comparisons


def speedup(batched: float, tensor: float) -> float:
    if tensor == 0.0:
        if batched == 0.0:
            return 1.0
        return math.inf
    return batched / tensor


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    stage_timings: Path,
    comparisons: list[dict[str, object]],
    summaries: list[dict[str, object]],
    top_n: int,
) -> None:
    comparison_by_key = {
        (
            str(row["toy"]),
            str(row["device"]),
            int(row["agent_count"]),
            str(row["stage"]),
        ): row
        for row in comparisons
    }
    steady_rows = [
        row for row in comparisons if row["stage"] == VIRTUAL_STEADY_STAGE
    ]
    tensor_focus = [
        row
        for row in summaries
        if row["training_backend"] == "tensor_batched"
        and row["stage"] in FOCUS_STAGES
        and float(row["seconds_per_epoch_mean"]) > 0.0
    ]
    cases = sorted(
        {
            (str(row["toy"]), str(row["device"]), int(row["agent_count"]))
            for row in tensor_focus
        }
    )

    lines = [
        "# Toy steady-state timing analysis",
        "",
        f"Input: `{stage_timings}`",
        "",
        (
            "`steady_state_total` is `sample_revision_mask + hooked_step_total`; "
            "setup, initial aggregate, per-epoch aggregate, and write stages are "
            "excluded. Child stages are nested diagnostics and should not be summed."
        ),
        "",
        "## Steady-state backend speedup",
        "",
        "| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        steady_rows,
        key=lambda item: (item["toy"], item["device"], int(item["agent_count"])),
    ):
        lines.append(
            "| {toy} | {device} | {agent_count} | {batched:.3f} | "
            "{tensor:.3f} | {speedup:.3f}x |".format(
                toy=row["toy"],
                device=row["device"],
                agent_count=row["agent_count"],
                batched=1000.0 * float(row["batched_seconds_per_epoch"]),
                tensor=1000.0 * float(row["tensor_batched_seconds_per_epoch"]),
                speedup=float(row["speedup"]),
            )
        )

    lines.extend(
        [
            "",
            "## Top tensor_batched focus stages",
            "",
            (
                "| toy | device | agents | stage | tensor ms/epoch | "
                "hook share | stage speedup |"
            ),
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for case in cases:
        case_rows = [
            row
            for row in tensor_focus
            if (row["toy"], row["device"], int(row["agent_count"])) == case
        ]
        case_rows.sort(key=lambda row: float(row["seconds_per_epoch_mean"]), reverse=True)
        for row in case_rows[:top_n]:
            comparison = comparison_by_key.get((*case, str(row["stage"])))
            stage_speedup = float(comparison["speedup"]) if comparison else math.nan
            lines.append(
                "| {toy} | {device} | {agent_count} | {stage} | {ms:.3f} | "
                "{share:.1f}% | {speedup:.3f}x |".format(
                    toy=row["toy"],
                    device=row["device"],
                    agent_count=row["agent_count"],
                    stage=row["stage"],
                    ms=1000.0 * float(row["seconds_per_epoch_mean"]),
                    share=float(row["share_of_hooked_step_pct"]),
                    speedup=stage_speedup,
                )
            )

    slower_rows = [
        row
        for row in comparisons
        if row["stage"] in FOCUS_STAGES
        and float(row["speedup"]) < 1.0
        and float(row["tensor_batched_seconds_per_epoch"]) > 0.0001
    ]
    slower_rows.sort(
        key=lambda row: float(row["tensor_minus_batched_seconds_per_epoch"]),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Tensor stages slower than batched",
            "",
            "| toy | device | agents | stage | delta ms/epoch | stage speedup |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in slower_rows[: max(top_n * 2, 1)]:
        lines.append(
            "| {toy} | {device} | {agent_count} | {stage} | {delta:.3f} | "
            "{speedup:.3f}x |".format(
                toy=row["toy"],
                device=row["device"],
                agent_count=row["agent_count"],
                stage=row["stage"],
                delta=1000.0 * float(row["tensor_minus_batched_seconds_per_epoch"]),
                speedup=float(row["speedup"]),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    totals, epoch_counts, stages = read_stage_timings(args.stage_timings)
    summaries = build_stage_summary(totals, epoch_counts, stages)
    comparisons = build_backend_comparison(summaries)

    write_csv(args.stage_summary_output, SUMMARY_FIELDS, summaries)
    write_csv(args.backend_comparison_output, COMPARISON_FIELDS, comparisons)
    if args.markdown_output is not None:
        write_markdown(
            args.markdown_output,
            stage_timings=args.stage_timings,
            comparisons=comparisons,
            summaries=summaries,
            top_n=args.top_n,
        )
    print(args.stage_summary_output)
    print(args.backend_comparison_output)
    if args.markdown_output is not None:
        print(args.markdown_output)


if __name__ == "__main__":
    main()
