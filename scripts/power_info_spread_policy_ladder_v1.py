"""Information-spread policy ladder, stage D2: the closed-loop rule on controls.

Specification: "D2 specification" in
docs/research/info-spread-policy-ladder-design-v1.md, fixed before this runs.
The rule is applied where the answer is known:

- NEG: two precomputed tables from disjoint donor sets, so no closed loop;
- POS-flat and POS-flip: a per-replicate shift in the veracity relation that
  only the adaptive arm can see in its own run.

No study cell is scored. L0 donors run only to build tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


CONTROLS: dict[str, float | None] = {"NEG_two_tables": None, "POS_flat": 0.0, "POS_flip": -3.0}
FAMILIES = ("linear", "mlp")
REGIMES = ("A", "B")
DONOR_SETS = (tuple(range(1000, 1004)), tuple(range(2000, 2004)))
SIZES = {"agents": 1000, "steps": 60, "items_per_step": 20, "epsilon": 0.02}
CELL_SEED_OFFSET = 1000
DELTA = 0.02
AGREEING_REPLICATES = 12
BOOTSTRAP_RESAMPLES = 2000
TAG_D2 = 2


def _learner(world: spread.World, family: str, cell: int, replicate: int) -> spread.Policy:
    config = world.config
    components = (TAG_D2, cell, ord(config.regime), spread.FAMILIES_INDEX[family], replicate)
    prior = spread.hand_prior(config)
    if config.regime == "A":
        return spread.pooled_learning_policy(family, prior, seed_components=components)
    return spread.per_agent_learning_policy(family, prior, config.agents, seed_components=components)


def _metrics(result: spread.RunResult) -> dict[str, float]:
    return {
        "share_precision": result.behavior["share_precision"],
        "false_reach": result.behavior["false_reach"],
        "seconds": round(result.seconds, 3),
    }


def run_cell(
    control: str, regime: str, family: str, cell: int, *,
    replicates: int = 16, sizes: Mapping[str, Any] = SIZES,
) -> dict[str, Any]:
    config = spread.WorldConfig(
        **sizes, regime=regime, geometry="linear", world_seed=CELL_SEED_OFFSET + cell,
        shift_credibility_weight=CONTROLS[control],
    )
    world = spread.build_world(config)

    def donors(seeds: Sequence[int]) -> list[core.PolicySnapshots]:
        return [
            spread.simulate(world, spread.build_stream(world, r), _learner(world, family, cell, r), r).snapshots
            for r in seeds
        ]

    first_donors = donors(DONOR_SETS[0])
    second_donors = donors(DONOR_SETS[1]) if control == "NEG_two_tables" else None
    rows = []
    for replicate in range(replicates):
        stream = spread.build_stream(world, replicate)
        if second_donors is not None:
            first = spread.table_policy("T0_traj_set1", first_donors, final_only=False)
            second = spread.table_policy("T0_traj_set2", second_donors, final_only=False)
        else:
            first = _learner(world, family, cell, replicate)
            second = spread.table_policy("T0_traj", first_donors, final_only=False)
        rows.append({
            "replicate": replicate,
            "early_shock": stream.early_shock(),
            "shift_step": stream.shift_step,
            "first": _metrics(spread.simulate(world, stream, first, replicate)),
            "second": _metrics(spread.simulate(world, stream, second, replicate)),
        })
    return {"control": control, "regime": regime, "family": family, "cell": cell, "rows": rows}


def _task(args: tuple[str, str, str, int, int]) -> dict[str, Any]:
    control, regime, family, cell, replicates = args
    return run_cell(control, regime, family, cell, replicates=replicates)


# -- the rule ----------------------------------------------------------------


def paired_rule(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray([r["first"]["share_precision"] - r["second"]["share_precision"] for r in rows])
    up, down = int((deltas >= DELTA).sum()), int((deltas <= -DELTA).sum())
    return {"holds": max(up, down) >= AGREEING_REPLICATES, "up": up, "down": down,
            "median_delta": float(np.median(deltas))}


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    variance = float((centered * centered).sum())
    return float((centered * (y - y.mean())).sum() / variance) if variance > 0 else float("nan")


def path_rule(rows: Sequence[Mapping[str, Any]], key: Sequence[int]) -> dict[str, Any]:
    shock = np.asarray([r["early_shock"] for r in rows])
    first = np.asarray([r["first"]["false_reach"] for r in rows])
    second = np.asarray([r["second"]["false_reach"] for r in rows])
    rng = np.random.default_rng(core.keyed_seed(*key))
    differences = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, len(rows), len(rows))
        difference = _slope(shock[index], first[index]) - _slope(shock[index], second[index])
        if np.isfinite(difference):
            differences.append(difference)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"holds": bool(low > 0 or high < 0), "interval": [float(low), float(high)],
            "slope_difference": _slope(shock, first) - _slope(shock, second)}


def score_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    key = (TAG_D2, list(CONTROLS).index(cell["control"]), ord(cell["regime"]),
           spread.FAMILIES_INDEX[cell["family"]], cell["cell"])
    r1, r2 = paired_rule(cell["rows"]), path_rule(cell["rows"], key)
    return {"r1": r1, "r2": r2, "material": r1["holds"] and r2["holds"]}


def _upper_bound(count: int, n: int) -> float | None:
    return round(1 - 0.05 ** (1 / n), 3) if count == 0 else None


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(f'{cell["control"]}/{cell["regime"]}/{cell["family"]}', []).append(score_cell(cell))
    table = {}
    for key, scores in sorted(groups.items()):
        n = len(scores)
        material = sum(s["material"] for s in scores)
        table[key] = {
            "cells": n,
            "material": material,
            "material_zero_upper_bound_95": _upper_bound(material, n),
            "r1_only_rule": sum(s["r1"]["holds"] for s in scores),
            "r2_only_rule": sum(s["r2"]["holds"] for s in scores),
            "median_cell_delta_share_precision": float(np.median([s["r1"]["median_delta"] for s in scores])),
        }
    reading = {}
    for regime in REGIMES:
        for family in FAMILIES:
            neg = table.get(f"NEG_two_tables/{regime}/{family}")
            flip = table.get(f"POS_flip/{regime}/{family}")
            if neg and flip:
                reading[f"{regime}/{family}"] = {
                    "neg_false_positive_ok": neg["material"] <= neg["cells"] // 20,
                    "pos_flip_power_ok": flip["material"] >= int(0.8 * flip["cells"]),
                }
    return {"table": table, "reading": reading}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=16)
    args = parser.parse_args()
    # Heaviest groups first, for load balance.
    order = sorted(
        ((c, r, f) for c in CONTROLS for r in REGIMES for f in FAMILIES),
        key=lambda t: (t[1] != "B", t[2] != "mlp", t[0] == "NEG_two_tables"),
    )
    tasks = [(c, r, f, cell, args.replicates) for c, r, f in order for cell in range(args.cells)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.workers <= 1:
        cells = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cells = list(pool.map(_task, tasks))
    (args.output_dir / "cells.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in cells), encoding="utf-8"
    )
    summary = {
        "status": "D2_CONTROLS_ONLY_NO_STUDY_CELL_SCORED",
        "sizes": SIZES, "cells_per_group": args.cells, "replicates": args.replicates,
        "wall_seconds": round(time.perf_counter() - started, 1),
        **summarize(cells),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["reading"], indent=1))


if __name__ == "__main__":
    main()
