"""Information-spread policy ladder, transfer study v1 (O23).

Design: docs/research/info-spread-policy-ladder-transfer-v1.md
Contract: experiments/configs/info_spread_policy_ladder_transfer_v1.yaml

Study v2 found that a table written before the run suffices in stationary
worlds, but its donors ran in the same world as the evaluation: the same
agents, network, and costs. This study asks whether a table still suffices
when its donors ran in other worlds drawn from the same generator: the same
regime and veracity relation, but another network, other thresholds, and
other costs.

Two arms are new, in study v2's cells:
- T0x_1: the policy trajectory of one donor run in another world;
- T0x_4: the average of four donor runs, each in a different other world.
Every other arm (F0, F1, T0_1, T0_4, L0) is read from study v2's development
cells, bound by its record. At replicate 0 of every cell, L0 is recomputed
and must equal study v2's value.

The primary measure and the group rule are study v2's. A development run
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
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import info_spread_unverified_measure_v1 as measure  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_transfer_v1.yaml"
ARMS = ("T0x_1", "T0x_4")
COMPARISONS = {"transfer_closed_loop": ("L0", "T0x_1"), "transfer_best_table": ("L0", "T0x_4")}
DESCRIPTIVE = {"transfer_cost": ("T0_4", "T0x_4"), "transfer_cost_single": ("T0_1", "T0x_1")}
LABELS = ("POSITIVE", "NEGATIVE", "INDISTINGUISHABLE")


# -- contract --------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    if tuple(contract["arms"]) != ARMS:
        raise ValueError("arms differ from the runner's")
    if {k: tuple(v) for k, v in contract["comparisons"].items()} != COMPARISONS:
        raise ValueError("comparisons differ from the runner's")
    return contract


def load_v2_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return v2.load_contract(ROOT / contract["study_v2"]["contract"])


verify_source_bindings = v2.verify_source_bindings
verify_precommit = v2.verify_precommit
_sha256 = v2._sha256


# -- donor worlds ------------------------------------------------------------------------


def donor_world_seed(contract: Mapping[str, Any], cell: int, donor: int) -> int:
    spec = contract["donors"]
    return int(spec["world_seed_offset"]) + int(spec["count"]) * cell + donor


def donor_cell_id(contract: Mapping[str, Any], cell: int, donor: int) -> int:
    """The cell index that keys a donor learner's seeds; disjoint from study cells."""

    spec = contract["donors"]
    return int(spec["learner_cell_offset"]) + int(spec["count"]) * cell + donor


def build_donor_world(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, regime: str,
                      geometry: str, cell: int, donor: int) -> spread.World:
    sizes = v2_contract["profiles"][profile]
    spec = v2_contract["world"]
    return spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(spec["degree"]), rewire=float(spec["rewire"]), seeds_per_item=int(spec["seeds_per_item"]),
        delay=int(spec["delay"]), visibility=float(spec["visibility"]), epsilon=float(spec["epsilon"]),
        refit_every=int(spec["refit_every"]), regime=regime, geometry=geometry,
        world_seed=donor_world_seed(contract, cell, donor),
    ))


def run_cell(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, regime: str, family: str,
             geometry: str, cell: int, device: str) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    geometry_index = list(v2_contract["design"]["geometries"]).index(geometry)
    replicate = int(contract["donors"]["stream_replicate"])
    donors = []
    for donor in range(int(contract["donors"]["count"])):
        donor_world = build_donor_world(contract, v2_contract, profile, regime, geometry, cell, donor)
        learner = v2._learner(v2_contract, donor_world, family, geometry_index, donor_cell_id(contract, cell, donor),
                              replicate, device)
        donors.append(spread.simulate(donor_world, spread.build_stream(donor_world, replicate), learner,
                                      replicate).snapshots)
    rows = []
    for r in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, r)
        arms = {"T0x_1": spread.table_policy("T0x_1", donors[:1], final_only=False),
                "T0x_4": spread.table_policy("T0x_4", donors, final_only=False)}
        rows.append({"replicate": r, "arms": {
            name: v2.metrics(measure.measured_simulate(world, stream, policy, r)[1]) for name, policy in arms.items()
        }})
    stream = spread.build_stream(world, 0)
    parity = v2.metrics(measure.measured_simulate(
        world, stream, v2._learner(v2_contract, world, family, geometry_index, cell, 0, device), 0)[1])
    return {"regime": regime, "family": family, "geometry": geometry, "cell": cell, "rows": rows,
            "l0_replicate_0": parity}


# -- study v2 inputs ---------------------------------------------------------------------


def load_v2_cells(contract: Mapping[str, Any], cells_path: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """Study v2's development cells, bound by its record (a source binding)."""

    record = json.loads((ROOT / contract["study_v2"]["record"]).read_text(encoding="utf-8"))
    if not all(record["reproduction"]["byte_identical"].values()):
        raise RuntimeError("study v2 did not reproduce byte for byte")
    if _sha256(cells_path) != record["reproduction"]["outputs"]["development"]["cells.jsonl"]:
        raise RuntimeError("the study v2 cells differ from its record")
    cells = [json.loads(line) for line in cells_path.read_text(encoding="utf-8").splitlines()]
    return {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in cells}


# -- the registered rule -----------------------------------------------------------------


def _label(rule: Mapping[str, Any]) -> str:
    if not rule["material"]:
        return "INDISTINGUISHABLE"
    return "POSITIVE" if rule["mean_cell_delta"] > 0 else "NEGATIVE"


def summarize(contract: Mapping[str, Any], v2_contract: Mapping[str, Any],
              v2_cells: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
              cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parity_failures = [
        f'{c["regime"]}/{c["family"]}/{c["geometry"]}/{c["cell"]}' for c in cells
        if c["l0_replicate_0"] != v2_cells[(c["regime"], c["family"], c["geometry"], c["cell"])]["rows"][0]["arms"]["L0"]
    ]
    groups: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = {}
    for cell in sorted(cells, key=lambda c: (c["regime"], c["family"], c["geometry"], c["cell"])):
        base = v2_cells[(cell["regime"], cell["family"], cell["geometry"], cell["cell"])]
        rows = []
        for a, b in zip(base["rows"], cell["rows"], strict=True):
            if a["replicate"] != b["replicate"]:
                raise RuntimeError("replicates are not aligned")
            rows.append({"replicate": a["replicate"], "arms": {**a["arms"], **b["arms"]}})
        groups.setdefault((cell["regime"], cell["family"], cell["geometry"]), []).append(rows)
    tag = int(contract["seeds"]["rule_tag"])
    every = {**COMPARISONS, **DESCRIPTIVE}
    arms = (*v2.ARMS, *ARMS)
    table: dict[str, Any] = {}
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        for m_index, name in enumerate(v2.MEASURES):
            rules = {}
            for c_index, (comparison, (left, right)) in enumerate(every.items()):
                means = [float(np.mean([r["arms"][left][name] - r["arms"][right][name] for r in rows]))
                         for rows in members]
                if not np.all(np.isfinite(means)):
                    raise RuntimeError(f"a {name} is undefined in {regime}/{family}/{geometry}")
                rules[comparison] = v2._group_rule(contract, means, (tag, g_index, m_index, c_index))
            entry[name] = rules
            entry[f"mean_{name}"] = {
                arm: round(float(np.mean([r["arms"][arm][name] for rows in members for r in rows])), 4) for arm in arms
            }
        table[f"{regime}/{family}/{geometry}"] = entry
    primary = {key: entry[v2.PRIMARY] for key, entry in table.items()}
    return {"table": table, "parity_failures": parity_failures,
            "dispositions": dispositions(contract, v2_contract, primary)}


def dispositions(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    design = v2_contract["design"]
    thresholds = contract["dispositions"]
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    labels = {name: {k: _label(v[name]) for k, v in study.items()} for name in COMPARISONS}
    counts = {name: {regime: {label: sum(1 for k, v in by_group.items() if k.startswith(f"{regime}/") and v == label)
                              for label in LABELS} for regime in design["regimes"]}
              for name, by_group in labels.items()}
    primary = counts["transfer_best_table"]
    by_regime = {}
    for regime, c in primary.items():
        if c["POSITIVE"] >= int(thresholds["adaptive_at_least"]):
            by_regime[regime] = "ADAPTIVE_BEATS_TRANSFERRED_TABLE"
        elif c["POSITIVE"] <= int(thresholds["table_suffices_at_most"]):
            by_regime[regime] = "TRANSFERRED_TABLE_SUFFICES"
        else:
            by_regime[regime] = "MIXED"
    null_violations = [
        k for k, v in table.items()
        if k.split("/")[2] == design["null_geometry"]
        and any(v[name]["material"] for name in COMPARISONS)
    ]
    return {
        "design_check": "DESIGN_DEFECT_IN_NULL_WORLD" if null_violations else "NULL_WORLD_PASSES",
        "null_violations": null_violations,
        "transfer_best_table": by_regime,
        "counts": counts, "labels_by_group": labels,
        "study_groups_per_regime": len(study) // len(design["regimes"]),
    }


# -- main --------------------------------------------------------------------------------


def _task(args: tuple[str, str, int, str, str, str, int]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    contract_path, profile, index, regime, family, geometry, cell = args
    contract = load_contract(Path(contract_path))
    v2_contract = load_v2_contract(contract)
    v2.apply_learner_settings(v2_contract)
    device = v2.device_for(family)
    started = time.perf_counter()
    result = run_cell(contract, v2_contract, profile, regime, family, geometry, cell, device)
    return index, result, {"device": device, "task_seconds": round(time.perf_counter() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", choices=("development", "smoke"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--pilot-cells", type=int, default=None,
                        help="a full-load pilot with this many cells per group; never read for results")
    parser.add_argument("--v2-cells", type=Path, help="study v2's development cells.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    v2_contract = load_v2_contract(contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    if is_registered_run and args.v2_cells is None:
        parser.error("a registered run needs --v2-cells")
    design = v2_contract["design"]
    cells_per_group = args.pilot_cells or int(v2_contract["profiles"][args.profile]["cells"])
    specs = [(r, f, g, cell) for r in design["regimes"] for f in design["families"][r]
             for g in design["geometries"] for cell in range(cells_per_group)]
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
    summary: dict[str, Any] = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "cells_per_group": cells_per_group, "primary_measure": v2.PRIMARY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
    }
    if args.v2_cells is not None:
        summary.update(summarize(contract, v2_contract, load_v2_cells(contract, args.v2_cells), ordered))
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": v2._host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"dispositions": summary.get("dispositions", {}).get("transfer_best_table"),
                      "design_check": summary.get("dispositions", {}).get("design_check"),
                      "parity_failures": summary.get("parity_failures"), **receipt["outputs"]}))


if __name__ == "__main__":
    main()
