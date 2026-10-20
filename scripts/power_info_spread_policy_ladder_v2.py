"""Information-spread policy ladder, stage D2b: the revised rule on controls.

Specification: "D2b specification" in
docs/research/info-spread-policy-ladder-design-v1.md, fixed before this runs.

Changes from D2 (power_info_spread_policy_ladder_v1.py):
- public fact-checks: learners train on every decided exposure;
- the Pivot learner settings;
- partial pooling in Regime B;
- two comparators, T0-1 at matched evidence and T0-4 as the best table;
- the POS-sign control;
- a group-level rule.

No study cell is scored.
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

import diagnose_info_spread_d2_v1 as d2diag  # noqa: E402
import policy_ladder_core_v1 as core  # noqa: E402
import power_info_spread_policy_ladder_v1 as power  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


CONTROLS = ("NEG", "POS_sign", "POS_flip")
FAMILIES = {"A": ("linear", "mlp"), "B": ("partial_linear", "partial_mlp")}
PAIRS = {
    "NEG": (("T0_1a", "T0_1b"), ("T0_4a", "T0_4b")),
    "POS_sign": (("L0", "T0_1"), ("L0", "T0_4")),
    "POS_flip": (("L0", "T0_1"), ("L0", "T0_4"), ("L0_window", "T0_1"), ("L0_window", "L0")),
}
LABEL_MODE = "all"
WINDOW = 12
MIN_GROUP_EFFECT = 0.01
TAG_D2B = 4
CELL_SEED_OFFSET = 3000


def _family_index(family: str) -> int:
    return {"linear": 1, "mlp": 2, "partial_linear": 4, "partial_mlp": 5}[family]


def _learner(world: spread.World, family: str, cell: int, replicate: int, window: int | None = None) -> spread.Policy:
    config = world.config
    components = (TAG_D2B, cell, ord(config.regime), _family_index(family), replicate)
    prior = spread.hand_prior(config)
    if config.regime == "A":
        return spread.pooled_learning_policy(
            family, prior, seed_components=components, window=window, label_mode=LABEL_MODE
        )
    return spread.partial_pooling_policy(
        family.removeprefix("partial_"), prior, config.agents, seed_components=components,
        window=window, label_mode=LABEL_MODE,
    )


def _world(control: str, regime: str, cell: int, sizes: Mapping[str, Any]) -> spread.World:
    return spread.build_world(spread.WorldConfig(
        **sizes, regime=regime, geometry="linear", world_seed=CELL_SEED_OFFSET + cell,
        shift_credibility_weight=-3.0 if control == "POS_flip" else None,
        random_credibility_sign=control == "POS_sign",
    ))


def run_cell(
    control: str, regime: str, family: str, cell: int, *,
    replicates: int = 16, sizes: Mapping[str, Any] = power.SIZES,
) -> dict[str, Any]:
    world = _world(control, regime, cell, sizes)

    def donors(seeds: Sequence[int]) -> list[core.PolicySnapshots]:
        return [
            spread.simulate(world, spread.build_stream(world, r), _learner(world, family, cell, r), r).snapshots
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
                "L0": _learner(world, family, cell, replicate),
                "T0_1": spread.table_policy("T0_1", first[:1], final_only=False),
                "T0_4": spread.table_policy("T0_4", first, final_only=False),
            }
            if control == "POS_flip":
                arms["L0_window"] = _learner(world, family, cell, replicate, window=WINDOW)
        row: dict[str, Any] = {
            "replicate": replicate, "early_shock": stream.early_shock(),
            "shift_step": stream.shift_step, "credibility_sign": stream.credibility_sign, "arms": {},
        }
        for name, policy in arms.items():
            result = spread.simulate(world, stream, policy, replicate)
            metrics = {
                "share_precision": result.behavior["share_precision"],
                "false_reach": result.behavior["false_reach"],
                "seconds": round(result.seconds, 2),
            }
            if control == "POS_flip":
                metrics["cohort_precision"] = d2diag._cohort_precision(stream, result)
            row["arms"][name] = metrics
        rows.append(row)
    return {"control": control, "regime": regime, "family": family, "cell": cell, "rows": rows}


def _task(args: tuple[str, str, str, int, int]) -> dict[str, Any]:
    control, regime, family, cell, replicates = args
    spread.use_learner_settings("pivot")
    return run_cell(control, regime, family, cell, replicates=replicates)


# -- rules ---------------------------------------------------------------------


def _pair_rows(cell: Mapping[str, Any], left: str, right: str) -> list[dict[str, Any]]:
    return [
        {"early_shock": r["early_shock"], "first": r["arms"][left], "second": r["arms"][right]}
        for r in cell["rows"]
    ]


def group_rule(cell_means: Sequence[float], key: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(cell_means, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence(tuple(int(k) for k in key)))
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(power.BOOTSTRAP_RESAMPLES)]
    low, high = (float(q) for q in np.quantile(means, [0.025, 0.975]))
    mean = float(values.mean())
    return {
        "mean_cell_delta": round(mean, 4), "interval_95": [round(low, 4), round(high, 4)],
        "material": bool((low > 0 or high < 0) and abs(mean) >= MIN_GROUP_EFFECT),
    }


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["control"], cell["regime"], cell["family"]), []).append(cell)
    table: dict[str, Any] = {}
    for g_index, ((control, regime, family), members) in enumerate(sorted(groups.items())):
        for p_index, (left, right) in enumerate(PAIRS[control]):
            per_cell, means = [], []
            for c in members:
                rows = _pair_rows(c, left, right)
                key = (TAG_D2B, g_index, p_index, c["cell"])
                r1, r2 = power.paired_rule(rows), power.path_rule(rows, key)
                per_cell.append(r1["holds"] and r2["holds"])
                means.append(float(np.mean([
                    row["first"]["share_precision"] - row["second"]["share_precision"] for row in rows
                ])))
            entry = {
                "cells": len(members),
                "per_cell_material": int(sum(per_cell)),
                "group": group_rule(means, (TAG_D2B, 99, g_index, p_index)),
            }
            if control == "POS_flip":
                for cohort in ("pre_shift", "post_shift"):
                    cohort_means = [
                        float(np.nanmean([
                            r["arms"][left]["cohort_precision"][cohort] - r["arms"][right]["cohort_precision"][cohort]
                            for r in c["rows"]
                        ]))
                        for c in members
                    ]
                    entry[f"group_{cohort}"] = group_rule(cohort_means, (TAG_D2B, 98, g_index, p_index, len(cohort)))
            table[f"{control}/{regime}/{family}/{left}-{right}"] = entry
    reading = {
        "neg_no_material_pair": all(
            not v["group"]["material"] for k, v in table.items() if k.startswith("NEG/")
        ),
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=16)
    args = parser.parse_args()
    groups = [(c, r, f) for c in CONTROLS for r in FAMILIES for f in FAMILIES[r]]
    groups.sort(key=lambda g: (g[1] != "B", "mlp" not in g[2], g[0] == "NEG"))
    tasks = [(c, r, f, cell, args.replicates) for c, r, f in groups for cell in range(args.cells)]
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
        "status": "D2B_CONTROLS_ONLY_NO_STUDY_CELL_SCORED",
        "learner_settings": "pivot", "label_mode": LABEL_MODE, "window": WINDOW,
        "sizes": power.SIZES, "cells_per_group": args.cells, "replicates": args.replicates,
        "wall_seconds": round(time.perf_counter() - started, 1),
        **summarize(cells),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["reading"], indent=1))


if __name__ == "__main__":
    main()
