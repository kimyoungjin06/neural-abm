"""Information-spread policy ladder, stage D2d: the unverified-item measure on controls.

Specification: docs/research/info-spread-policy-ladder-d2d-v1.md, fixed
before this runs.

D4 found that an MLP learner recognizes items it has seen fact-checked, so
L0 beat F1 in a world with nothing to learn. The user chose on 2026-10-18 to
make the primary measure share precision on items whose fact-check is not
public at the decision (info_spread_unverified_measure_v1.py). D2d checks the
group rule on that measure:

- NEG, two tables from disjoint donor sets: no material pair;
- POS-sign, a run-specific sign of the credibility relation: L0 - T0-1
  material and positive in every group;
- NULL, the null world: L0 - F1 not material and positive in any group.
  On the old measure, NULL should show D4's defect again, which shows the
  control can detect it.

Learners, observability, sizes, and the rule are D3a's. No study cell is
scored.
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

import info_spread_unverified_measure_v1 as measure  # noqa: E402
import policy_ladder_core_v1 as core  # noqa: E402
import power_info_spread_policy_ladder_v1 as power  # noqa: E402
import power_info_spread_policy_ladder_v2 as power2  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as d4  # noqa: E402


CONTROLS = ("NEG", "POS_sign", "NULL")
FAMILIES = {"A": ("linear_exact", "mlp"), "B": ("partial_linear_exact", "partial_mlp")}
PAIRS = {
    "NEG": (("T0_1a", "T0_1b"), ("T0_4a", "T0_4b")),
    "POS_sign": (("L0", "T0_1"), ("L0", "T0_4")),
    "NULL": (("L0", "F1"), ("L0", "T0_1"), ("L0", "T0_4")),
}
MEASURES = ("unverified_share_precision", "share_precision")
TAG_D2D = 8
CELL_SEED_OFFSET = 5000


def _learner(d4_contract: Mapping[str, Any], world: spread.World, family: str, control: str, cell: int,
             replicate: int, device: str) -> spread.Policy:
    """D3a's learner, with D2d's own seed components."""

    config = world.config
    components = (TAG_D2D, CONTROLS.index(control), cell, ord(config.regime), d4._family_index(family), replicate)
    prior = spread.hand_prior(config)
    observability = d4_contract["world"]["observability"]
    one_row = bool(observability["one_row_per_agent_and_item"])
    if config.regime == "A":
        return spread.pooled_learning_policy(
            family, prior, seed_components=components, label_mode=observability["label_mode"],
            one_row_per_exposure=one_row, device=device,
        )
    return spread.partial_pooling_policy(
        family.removeprefix("partial_"), prior, config.agents, seed_components=components,
        label_mode=observability["label_mode"], device=device, agent_residual="exact", one_row_per_exposure=one_row,
    )


def build_world(d4_contract: Mapping[str, Any], control: str, regime: str, cell: int,
                sizes: Mapping[str, Any]) -> spread.World:
    spec = d4_contract["world"]
    return spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(spec["degree"]), rewire=float(spec["rewire"]), seeds_per_item=int(spec["seeds_per_item"]),
        delay=int(spec["delay"]), visibility=float(spec["visibility"]), epsilon=float(spec["epsilon"]),
        refit_every=int(spec["refit_every"]), regime=regime, geometry="null" if control == "NULL" else "linear",
        world_seed=CELL_SEED_OFFSET + cell, random_credibility_sign=control == "POS_sign",
    ))


def _metrics(measured: Mapping[str, Any]) -> dict[str, Any]:
    return {key: measured[key] for key in (*MEASURES, "false_reach", "decisions", "by_state")}


def run_cell(d4_contract: Mapping[str, Any], control: str, regime: str, family: str, cell: int, *,
             replicates: int, sizes: Mapping[str, Any], device: str = "cpu") -> dict[str, Any]:
    world = build_world(d4_contract, control, regime, cell, sizes)

    def donors(seeds: Sequence[int]) -> list[core.PolicySnapshots]:
        return [
            spread.simulate(world, spread.build_stream(world, r),
                            _learner(d4_contract, world, family, control, cell, r, device), r).snapshots
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
                "L0": _learner(d4_contract, world, family, control, cell, replicate, device),
                "T0_1": spread.table_policy("T0_1", first[:1], final_only=False),
                "T0_4": spread.table_policy("T0_4", first, final_only=False),
            }
            if control == "NULL":
                arms["F1"] = spread.prior_policy("F1", spread.hand_prior(world.config))
        metrics = {name: _metrics(measure.measured_simulate(world, stream, policy, replicate)[1])
                   for name, policy in arms.items()}
        rows.append({"replicate": replicate, "credibility_sign": stream.credibility_sign, "arms": metrics})
    return {"control": control, "regime": regime, "family": family, "cell": cell, "device": device, "rows": rows}


def _task(args: tuple[str, str, str, int, int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    control, regime, family, cell, replicates, sizes = args
    d4_contract = d4.load_contract()
    d4.apply_learner_settings(d4_contract)
    device = d4.device_for(family)
    started = time.perf_counter()
    result = run_cell(d4_contract, control, regime, family, cell, replicates=replicates, sizes=sizes, device=device)
    return result, {"device": device, "task_seconds": round(time.perf_counter() - started, 1)}


# -- the group rule on both measures -------------------------------------------------


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["control"], cell["regime"], cell["family"]), []).append(cell)
    table: dict[str, Any] = {}
    for g_index, ((control, regime, family), members) in enumerate(sorted(groups.items())):
        members = sorted(members, key=lambda c: c["cell"])
        for p_index, (left, right) in enumerate(PAIRS[control]):
            entry: dict[str, Any] = {"cells": len(members)}
            for m_index, name in enumerate(MEASURES):
                means = [float(np.mean([r["arms"][left][name] - r["arms"][right][name] for r in c["rows"]]))
                         for c in members]
                if not np.all(np.isfinite(means)):
                    raise RuntimeError(f"a {name} is undefined in {control}/{regime}/{family}")
                entry[name] = power2.group_rule(means, (TAG_D2D, 99, g_index, p_index, m_index))
            table[f"{control}/{regime}/{family}/{left}-{right}"] = entry

    def rule(key: str, name: str = MEASURES[0]) -> Mapping[str, Any]:
        return table[key][name]

    reading = {
        "neg_no_material_pair": all(not rule(k)["material"] for k in table if k.startswith("NEG/")),
        "pos_sign_closed_loop_material_and_positive": all(
            rule(k)["material"] and rule(k)["mean_cell_delta"] > 0
            for k in table if k.startswith("POS_sign/") and k.endswith("/L0-T0_1")
        ),
        "null_no_material_positive_learner_gain": not any(
            rule(k)["material"] and rule(k)["mean_cell_delta"] > 0
            for k in table if k.startswith("NULL/") and k.endswith("/L0-F1")
        ),
    }
    reading["adequate"] = all(reading.values())
    reading["null_old_measure_material_positive_groups"] = sorted(
        k.removesuffix("/L0-F1") for k in table
        if k.startswith("NULL/") and k.endswith("/L0-F1")
        and rule(k, "share_precision")["material"] and rule(k, "share_precision")["mean_cell_delta"] > 0
    )
    return {"table": table, "reading": reading}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=16)
    args = parser.parse_args()
    sizes = dict(power.SIZES)
    groups = [(c, r, f) for c in CONTROLS for r in FAMILIES for f in FAMILIES[r]]
    groups.sort(key=lambda g: ("mlp" not in g[2], g[0] == "NEG"))  # heaviest first
    tasks = [(c, r, f, cell, args.replicates, sizes) for c, r, f in groups for cell in range(args.cells)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, task) for task in tasks]
        for future in as_completed(futures):
            cell, timing = future.result()
            cells.append(cell)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "done": len(cells), "of": len(tasks),
                    "group": f'{cell["control"]}/{cell["regime"]}/{cell["family"]}', "cell": cell["cell"],
                    **timing, "elapsed_seconds": round(time.perf_counter() - started, 1),
                }) + "\n")
    ordered = sorted(cells, key=lambda c: (c["control"], c["regime"], c["family"], c["cell"]))
    (args.output_dir / "cells.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in ordered), encoding="utf-8")
    summary = {
        "status": "D2D_CONTROLS_ONLY_NO_STUDY_CELL_SCORED",
        "primary_measure": MEASURES[0], "sizes": sizes, "cell_seed_offset": CELL_SEED_OFFSET,
        "cells_per_group": args.cells, "replicates": args.replicates, "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        **summarize(ordered),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["reading"], indent=1))


if __name__ == "__main__":
    main()
