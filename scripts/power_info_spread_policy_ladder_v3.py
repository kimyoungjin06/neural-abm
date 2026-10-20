"""Information-spread policy ladder, stage D2c: the revised rule on controls.

Specification: "D2c specification" in
docs/research/info-spread-policy-ladder-design-v1.md, fixed before this runs.

Changes from D2b (power_info_spread_policy_ladder_v2.py):
- exact (Newton) linear learners, pooled and per agent;
- one training row per agent and item under public fact-checks;
- pooled MLPs trained on GPUs, 24 workers, per-task progress;
- the group rule alone.

No study cell is scored.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS needs this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_info_spread_d2_v1 as d2diag  # noqa: E402
import policy_ladder_core_v1 as core  # noqa: E402
import power_info_spread_policy_ladder_v1 as power  # noqa: E402
import power_info_spread_policy_ladder_v2 as power2  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


CONTROLS = power2.CONTROLS
FAMILIES = {"A": ("linear_exact", "mlp"), "B": ("partial_linear_exact", "partial_mlp")}
PAIRS = power2.PAIRS
LABEL_MODE = "all"
WINDOW = 12
TAG_D2C = 5
CELL_SEED_OFFSET = 4000


def _family_index(family: str) -> int:
    return {"linear_exact": 6, "mlp": 2, "partial_linear_exact": 7, "partial_mlp": 5}[family]


def _learner(
    world: spread.World, family: str, cell: int, replicate: int, device: str,
    window: int | None = None,
) -> spread.Policy:
    config = world.config
    components = (TAG_D2C, cell, ord(config.regime), _family_index(family), replicate)
    prior = spread.hand_prior(config)
    if config.regime == "A":
        return spread.pooled_learning_policy(
            family, prior, seed_components=components, window=window, label_mode=LABEL_MODE,
            one_row_per_exposure=True, device=device,
        )
    return spread.partial_pooling_policy(
        family.removeprefix("partial_"), prior, config.agents, seed_components=components,
        window=window, label_mode=LABEL_MODE, device=device, agent_residual="exact",
        one_row_per_exposure=True,
    )


def _world(control: str, regime: str, cell: int, sizes: Mapping[str, Any]) -> spread.World:
    return spread.build_world(spread.WorldConfig(
        **sizes, regime=regime, geometry="linear", world_seed=CELL_SEED_OFFSET + cell,
        shift_credibility_weight=-3.0 if control == "POS_flip" else None,
        random_credibility_sign=control == "POS_sign",
    ))


def device_for(task_index: int, family: str) -> str:
    """Pooled MLPs train on a GPU when one is visible; linear fits stay on CPU."""

    import torch

    if "mlp" not in family or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{task_index % torch.cuda.device_count()}"


def run_cell(
    control: str, regime: str, family: str, cell: int, *,
    replicates: int = 16, sizes: Mapping[str, Any] = power.SIZES, device: str = "cpu",
) -> dict[str, Any]:
    world = _world(control, regime, cell, sizes)

    def donors(seeds: Sequence[int]) -> list[core.PolicySnapshots]:
        return [
            spread.simulate(world, spread.build_stream(world, r), _learner(world, family, cell, r, device), r).snapshots
            for r in seeds
        ]

    first = donors(power.DONOR_SETS[0])
    second = donors(power.DONOR_SETS[1]) if control == "NEG" else None
    rows = []
    for replicate in range(replicates):
        stream = spread.build_stream(world, replicate)
        if control == "NEG":
            arms = {
                "T0_1a": spread.table_policy("T0_1a", first[:1], final_only=False),
                "T0_1b": spread.table_policy("T0_1b", second[:1], final_only=False),
                "T0_4a": spread.table_policy("T0_4a", first, final_only=False),
                "T0_4b": spread.table_policy("T0_4b", second, final_only=False),
            }
        else:
            arms = {
                "L0": _learner(world, family, cell, replicate, device),
                "T0_1": spread.table_policy("T0_1", first[:1], final_only=False),
                "T0_4": spread.table_policy("T0_4", first, final_only=False),
            }
            if control == "POS_flip":
                arms["L0_window"] = _learner(world, family, cell, replicate, device, window=WINDOW)
        row: dict[str, Any] = {
            "replicate": replicate, "shift_step": stream.shift_step,
            "credibility_sign": stream.credibility_sign, "arms": {},
        }
        for name, policy in arms.items():
            result = spread.simulate(world, stream, policy, replicate)
            metrics = {"share_precision": result.behavior["share_precision"], "seconds": round(result.seconds, 2)}
            if control == "POS_flip":
                metrics["cohort_precision"] = d2diag._cohort_precision(stream, result)
            row["arms"][name] = metrics
        rows.append(row)
    return {"control": control, "regime": regime, "family": family, "cell": cell, "device": device, "rows": rows}


def _task(args: tuple[int, str, str, str, int, int]) -> dict[str, Any]:
    index, control, regime, family, cell, replicates = args
    spread.use_learner_settings("pivot")
    started = time.perf_counter()
    result = run_cell(control, regime, family, cell, replicates=replicates, device=device_for(index, family))
    result["task_seconds"] = round(time.perf_counter() - started, 1)
    return result


# -- the group rule --------------------------------------------------------------


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["control"], cell["regime"], cell["family"]), []).append(cell)
    table: dict[str, Any] = {}
    for g_index, ((control, regime, family), members) in enumerate(sorted(groups.items())):
        for p_index, (left, right) in enumerate(PAIRS[control]):
            means = [
                float(np.mean([r["arms"][left]["share_precision"] - r["arms"][right]["share_precision"] for r in c["rows"]]))
                for c in members
            ]
            entry = {"cells": len(members), "group": power2.group_rule(means, (TAG_D2C, 99, g_index, p_index))}
            if control == "POS_flip":
                for cohort in ("pre_shift", "post_shift"):
                    cohort_means = [
                        float(np.nanmean([
                            r["arms"][left]["cohort_precision"][cohort] - r["arms"][right]["cohort_precision"][cohort]
                            for r in c["rows"]
                        ]))
                        for c in members
                    ]
                    entry[f"group_{cohort}"] = power2.group_rule(cohort_means, (TAG_D2C, 98, g_index, p_index, len(cohort)))
            table[f"{control}/{regime}/{family}/{left}-{right}"] = entry
    reading = {
        "neg_no_material_pair": all(not v["group"]["material"] for k, v in table.items() if k.startswith("NEG/")),
        "pos_sign_closed_loop_material_and_positive": all(
            v["group"]["material"] and v["group"]["mean_cell_delta"] > 0
            for k, v in table.items() if k.startswith("POS_sign/") and k.endswith("/L0-T0_1")
        ),
    }
    reading["adequate"] = reading["neg_no_material_pair"] and reading["pos_sign_closed_loop_material_and_positive"]
    return {"table": table, "reading": reading}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=16)
    args = parser.parse_args()
    groups = [(c, r, f) for c in CONTROLS for r in FAMILIES for f in FAMILIES[r]]
    groups.sort(key=lambda g: ("mlp" not in g[2], g[0] == "NEG"))
    specs = [(c, r, f, cell) for c, r, f in groups for cell in range(args.cells)]
    tasks = [(i, c, r, f, cell, args.replicates) for i, (c, r, f, cell) in enumerate(specs)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, task): task[0] for task in tasks}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            cell = results[index]
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "done": len(results), "of": len(tasks), "task": index,
                    "group": f'{cell["control"]}/{cell["regime"]}/{cell["family"]}', "cell": cell["cell"],
                    "device": cell["device"], "task_seconds": cell["task_seconds"],
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }) + "\n")
    cells = [results[i] for i in range(len(tasks))]
    (args.output_dir / "cells.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in cells), encoding="utf-8"
    )
    summary = {
        "status": "D2C_CONTROLS_ONLY_NO_STUDY_CELL_SCORED",
        "learner_settings": "pivot", "label_mode": LABEL_MODE, "one_row_per_exposure": True,
        "window": WINDOW, "sizes": power.SIZES, "cells_per_group": args.cells,
        "replicates": args.replicates, "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        **summarize(cells),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["reading"], indent=1))


if __name__ == "__main__":
    main()
