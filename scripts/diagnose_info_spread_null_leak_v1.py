"""Diagnosis of D4's null-world design defect (exploratory, not registered).

D4 found L0 - F1 material and positive in B/partial_mlp/null: in a world
where features carry no information about veracity, the learner shares
true items more often than the base rate. This asks where that gain arises.

Each policy's own share decisions (probability at least the cutoff, before
the exploration floor) are split by the state of the item's fact-check at the
decision:
- unrevealed: the item is never fact-checked;
- pending: it will be, but the check is not yet visible;
- visible: its check is already public.

A gain confined to visible items means the learner uses the item's own
fact-check. Two routes are possible:
- (a) the item's features identify it, and an MLP can memorize them;
- (b) in Regime B, the trust feature of the item's sharers moves with that
  item's own fact-check.
The variant L0_no_trust holds the trust feature at a constant, both in its
decisions and in its training rows, to separate (b) from (a).

Only null cells are run; no study cell is read.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as d4  # noqa: E402


GROUPS = (("A", "mlp"), ("B", "partial_mlp"), ("B", "partial_linear_exact"))
STATES = ("unrevealed", "pending", "visible")
TRUST_COLUMN = 6
TRUST_CONSTANT = 0.5


def _masked_record(record: core.OutcomeRecord) -> core.OutcomeRecord:
    masked = core.OutcomeRecord(width=record.width)
    for row in record.features:
        row = row.copy()
        row[TRUST_COLUMN] = TRUST_CONSTANT
        masked.features.append(row)
    masked.labels, masked.agents = list(record.labels), list(record.agents)
    masked.visible_from, masked.kinds = list(record.visible_from), list(record.kinds)
    return masked


def without_trust(policy: spread.Policy) -> spread.Policy:
    """The policy with the trust feature held constant in decisions and training."""

    inner, observe = policy.probability, policy.observe

    def probability(ids: np.ndarray, x: np.ndarray, step: int) -> np.ndarray:
        x = x.copy()
        x[:, TRUST_COLUMN] = TRUST_CONSTANT
        return inner(ids, x, step)

    policy.probability = probability
    if observe is not None:
        policy.observe = lambda record, step: observe(_masked_record(record), step)
    return policy


def logged(policy: spread.Policy, world: spread.World, stream: spread.Stream, log: list[np.ndarray]) -> spread.Policy:
    """Record (step, item, agent, policy share) for every decision."""

    index = {tuple(row): i for i, row in enumerate(stream.features)}
    inner = policy.probability

    def probability(ids: np.ndarray, x: np.ndarray, step: int) -> np.ndarray:
        p = inner(ids, x, step)
        items = np.asarray([index[tuple(row)] for row in x[:, :4]], dtype=np.int64)
        share = p >= world.share_cutoff[ids]
        log.append(np.stack([np.full(len(ids), step), items, ids, share.astype(np.int64)], axis=1))
        return p

    policy.probability = probability
    return policy


def split_precision(world: spread.World, stream: spread.Stream, log: list[np.ndarray]) -> dict[str, Any]:
    rows = np.vstack(log) if log else np.zeros((0, 4), dtype=np.int64)
    step, item, share = rows[:, 0], rows[:, 1], rows[:, 3].astype(bool)
    visible_at = stream.born[item] + world.config.delay
    state = np.where(~stream.revealed[item], 0, np.where(visible_at > step, 1, 2))
    out = {}
    for code, name in enumerate(STATES):
        chosen = share & (state == code)
        out[name] = {"shares": int(chosen.sum()),
                     "true": int(stream.veracity[item[chosen]].sum()),
                     "decisions": int((state == code).sum())}
    return out


def run_cell(d4_contract: Mapping[str, Any], regime: str, family: str, cell: int, replicates: int,
             device: str) -> dict[str, Any]:
    sizes = d4_contract["profiles"]["development"]
    spec = d4_contract["world"]
    world = spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(spec["degree"]), rewire=float(spec["rewire"]), seeds_per_item=int(spec["seeds_per_item"]),
        delay=int(spec["delay"]), visibility=float(spec["visibility"]), epsilon=float(spec["epsilon"]),
        refit_every=int(spec["refit_every"]), regime=regime, geometry="null",
        world_seed=int(d4_contract["seeds"]["world_seed_offset"]) + cell,
    ))
    geometry_index = list(d4_contract["design"]["geometries"]).index("null")
    rows = []
    for replicate in range(replicates):
        stream = spread.build_stream(world, replicate)
        arms = {
            "F1": spread.prior_policy("F1", spread.hand_prior(world.config)),
            "L0": d4._learner(d4_contract, world, family, geometry_index, cell, replicate, device),
        }
        if regime == "B" and "mlp" in family:
            arms["L0_no_trust"] = without_trust(
                d4._learner(d4_contract, world, family, geometry_index, cell, replicate, device))
        metrics = {}
        for name, policy in arms.items():
            log: list[np.ndarray] = []
            result = spread.simulate(world, stream, logged(policy, world, stream, log), replicate)
            metrics[name] = {"share_precision": result.behavior["share_precision"],
                             "split": split_precision(world, stream, log)}
        rows.append({"replicate": replicate, "arms": metrics})
    return {"regime": regime, "family": family, "cell": cell, "rows": rows}


def _task(args: tuple[str, str, int, int]) -> dict[str, Any]:
    regime, family, cell, replicates = args
    d4_contract = d4.load_contract()
    d4.apply_learner_settings(d4_contract)
    started = time.perf_counter()
    result = run_cell(d4_contract, regime, family, cell, replicates, d4.device_for(family))
    result["task_seconds"] = round(time.perf_counter() - started, 1)
    return result


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime, family in GROUPS:
        members = [c for c in cells if c["regime"] == regime and c["family"] == family]
        arms = sorted({a for c in members for r in c["rows"] for a in r["arms"]})
        entry: dict[str, Any] = {"cells": len(members), "runs": sum(len(c["rows"]) for c in members)}
        for arm in arms:
            runs = [r["arms"][arm] for c in members for r in c["rows"]]
            pooled = {}
            for state in STATES:
                shares = sum(r["split"][state]["shares"] for r in runs)
                true = sum(r["split"][state]["true"] for r in runs)
                pooled[state] = {"shares": shares, "precision": round(true / shares, 4) if shares else None}
            entry[arm] = {"share_precision": round(float(np.mean([r["share_precision"] for r in runs])), 4),
                          "policy_shares_by_fact_check_state": pooled}
        out[f"{regime}/{family}/null"] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    tasks = [(r, f, cell, args.replicates) for r, f in GROUPS for cell in range(args.cells)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = list(pool.map(_task, tasks))
    (args.output_dir / "cells.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cells))
    summary = {"status": "EXPLORATORY_DIAGNOSIS_NULL_CELLS_ONLY", "cells": args.cells,
               "replicates": args.replicates, "wall_seconds": round(time.perf_counter() - started, 1),
               "groups": summarize(cells)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["groups"], indent=1))


if __name__ == "__main__":
    main()
