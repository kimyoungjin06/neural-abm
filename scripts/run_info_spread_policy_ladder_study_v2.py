"""Information-spread policy ladder, study v2 (stage D4 v2 runner for D3a v2).

Design: docs/research/info-spread-policy-ladder-study-v2.md
Contract: experiments/configs/info_spread_policy_ladder_study_v2.yaml

Study v1 failed its null check: an MLP learner recognizes items it has seen
fact-checked. Study v2 keeps v1's world, arms, learners, and rule, and
changes three things:

- the primary measure is share precision on unverified items, whose
  fact-check is not public at the decision (info_spread_unverified_measure_v1);
- the null check applies to that measure;
- the world seeds are fresh.

Share precision over all items is recorded beside it. A development run
needs the precommit SHA; smoke and pilot runs skip that check and are never
read for results. cells.jsonl and summary.json carry no timing.
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
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as v1  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_study_v2.yaml"
ARMS = v1.ARMS
COMPARISONS = v1.COMPARISONS
MEASURES = ("unverified_share_precision", "share_precision")
PRIMARY = MEASURES[0]

# Unchanged from v1, re-exported for the LLM-arms runner.
verify_source_bindings = v1.verify_source_bindings
verify_precommit = v1.verify_precommit
apply_learner_settings = v1.apply_learner_settings
device_for = v1.device_for
_family_index = v1._family_index
_learner = v1._learner
_group_rule = v1._group_rule
_host = v1._host
_sha256 = v1._sha256


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = v1.load_contract(path)
    if contract["measures"]["primary"] != PRIMARY:
        raise ValueError("the primary measure differs from the runner's")
    return contract


def build_world(contract: Mapping[str, Any], profile: str, regime: str, geometry: str, cell: int) -> spread.World:
    sizes = contract["profiles"][profile]
    spec = contract["world"]
    return spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(spec["degree"]), rewire=float(spec["rewire"]), seeds_per_item=int(spec["seeds_per_item"]),
        delay=int(spec["delay"]), visibility=float(spec["visibility"]), epsilon=float(spec["epsilon"]),
        refit_every=int(spec["refit_every"]), regime=regime, geometry=geometry,
        world_seed=int(contract["seeds"]["world_seed_offset"]) + cell,
    ))


def metrics(measured: Mapping[str, Any]) -> dict[str, Any]:
    return {key: measured[key] for key in (*MEASURES, "false_reach", "decisions", "by_state")}


def run_cell(contract: Mapping[str, Any], profile: str, regime: str, family: str, geometry: str,
             cell: int, device: str) -> dict[str, Any]:
    world = build_world(contract, profile, regime, geometry, cell)
    geometry_index = list(contract["design"]["geometries"]).index(geometry)
    donors = [
        spread.simulate(world, spread.build_stream(world, r),
                        _learner(contract, world, family, geometry_index, cell, r, device), r).snapshots
        for r in contract["seeds"]["donor_replicates"]
    ]
    prior = spread.hand_prior(world.config)
    rows = []
    for replicate in range(int(contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        arms = {
            "F0": measure.threshold_probability_policy(world),
            "F1": spread.prior_policy("F1", prior),
            "T0_1": spread.table_policy("T0_1", donors[:1], final_only=False),
            "T0_4": spread.table_policy("T0_4", donors, final_only=False),
            "L0": _learner(contract, world, family, geometry_index, cell, replicate, device),
        }
        rows.append({"replicate": replicate, "arms": {
            name: metrics(measure.measured_simulate(world, stream, policy, replicate)[1])
            for name, policy in arms.items()
        }})
    return {"regime": regime, "family": family, "geometry": geometry, "cell": cell, "rows": rows}


def _task(args: tuple[str, str, int, str, str, str, int]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    contract_path, profile, index, regime, family, geometry, cell = args
    contract = load_contract(Path(contract_path))
    apply_learner_settings(contract)
    device = device_for(family)
    started = time.perf_counter()
    result = run_cell(contract, profile, regime, family, geometry, cell, device)
    return index, result, {"device": device, "task_seconds": round(time.perf_counter() - started, 1)}


# -- the registered rule ---------------------------------------------------------------


def summarize(contract: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["regime"], cell["family"], cell["geometry"]), []).append(cell)
    table: dict[str, Any] = {}
    tag = int(contract["seeds"]["rule_tag"])
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        for m_index, name in enumerate(MEASURES):
            rules = {}
            for c_index, (comparison, (left, right)) in enumerate(COMPARISONS.items()):
                means = [float(np.mean([r["arms"][left][name] - r["arms"][right][name] for r in c["rows"]]))
                         for c in members]
                if not np.all(np.isfinite(means)):
                    raise RuntimeError(f"a {name} is undefined in {regime}/{family}/{geometry}")
                rules[comparison] = _group_rule(contract, means, (tag, g_index, m_index, c_index))
            entry[name] = rules
            entry[f"mean_{name}"] = {
                arm: round(float(np.mean([r["arms"][arm][name] for c in members for r in c["rows"]])), 4)
                for arm in ARMS
            }
        entry["mean_false_reach"] = {
            arm: round(float(np.mean([r["arms"][arm]["false_reach"] for c in members for r in c["rows"]])), 2)
            for arm in ARMS
        }
        table[f"{regime}/{family}/{geometry}"] = entry
    primary = {key: entry[PRIMARY] for key, entry in table.items()}
    return {"table": table, "dispositions": v1.dispositions(contract, primary)}


dispositions = v1.dispositions


# -- main ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", choices=("development", "smoke"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--pilot-cells", type=int, default=None,
                        help="a full-load pilot with this many cells per group; never read for results")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    design = contract["design"]
    cells_per_group = args.pilot_cells or int(contract["profiles"][args.profile]["cells"])
    specs = [
        (r, f, g, cell)
        for r in design["regimes"] for f in design["families"][r] for g in design["geometries"]
        for cell in range(cells_per_group)
    ]
    specs.sort(key=lambda s: "mlp" not in s[1])  # heaviest first, for load balance
    tasks = [(str(args.contract), args.profile, i, *spec) for i, spec in enumerate(specs)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, task) for task in tasks]
        for future in as_completed(futures):
            index, cell, timing = future.result()
            results[index] = cell
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "done": len(results), "of": len(tasks), "task": index,
                    "group": f'{cell["regime"]}/{cell["family"]}/{cell["geometry"]}', "cell": cell["cell"],
                    **timing, "elapsed_seconds": round(time.perf_counter() - started, 1),
                }) + "\n")
    ordered = sorted(results.values(), key=lambda c: (c["regime"], c["family"], c["geometry"], c["cell"]))
    cells_path = args.output_dir / "cells.jsonl"
    cells_path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in ordered), encoding="utf-8")
    summary = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "cells_per_group": cells_per_group, "primary_measure": PRIMARY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        **summarize(contract, ordered),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": _host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"dispositions": summary["dispositions"]["closed_loop"],
                      "design_check": summary["dispositions"]["design_check"], **receipt["outputs"]}))


if __name__ == "__main__":
    main()
