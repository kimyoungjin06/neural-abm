"""Information-spread policy ladder, flip-rate study v1 (O26).

Design: docs/research/info-spread-policy-ladder-flip-rate-v1.md
Contract: experiments/configs/info_spread_policy_ladder_flip_rate_v1.yaml

Study v2 found that, in stationary worlds, a table written before the run
suffices. The POS-sign control (D2c, D2d) found the opposite when each run
draws its own sign for the credibility relation: the closed loop beats both
tables. This study traces the region between the two.

Each run's credibility relation keeps its usual sign, or is reversed with
probability p, the flip rate. Its donors draw their signs the same way.

- At p = 0 every run is stationary and alike, as in study v2's linear world.
- At p = 0.5 each run's sign is a coin flip, as in the POS-sign control.

The study runs p in {0, 0.1, 0.25, 0.5} and asks at which flip rate the
adaptive learner starts to beat the best table written before the run.

The stream for a flip rate is built here, not in the frozen prototype. It
reproduces the prototype's stationary stream at p = 0 and its POS-sign
stream at p = 0.5, as tests check. A development run needs the precommit
SHA; smoke and pilot runs skip that check and are never read for results.
cells.jsonl and summary.json carry no timing.
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
import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_flip_rate_v1.yaml"
ARMS = ("F1", "T0_1", "T0_4", "L0")
COMPARISONS = {"closed_loop": ("L0", "T0_1"), "best_table": ("L0", "T0_4"), "hand_rule": ("L0", "F1")}
LABELS = ("POSITIVE", "NEGATIVE", "INDISTINGUISHABLE")
GEOMETRY = "linear"


# -- contract --------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    if tuple(contract["arms"]) != ARMS:
        raise ValueError("arms differ from the runner's")
    if {k: tuple(v) for k, v in contract["comparisons"].items()} != COMPARISONS:
        raise ValueError("comparisons differ from the runner's")
    rates = [float(p) for p in contract["design"]["flip_rates"]]
    if rates != sorted(rates) or rates[0] != 0.0 or rates[-1] != 0.5:
        raise ValueError("flip rates must rise from 0 to 0.5")
    return contract


def load_v2_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return v2.load_contract(ROOT / contract["study_v2"]["contract"])


verify_source_bindings = v2.verify_source_bindings
verify_precommit = v2.verify_precommit
_sha256 = v2._sha256


# -- worlds and streams -----------------------------------------------------------------


def build_world(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, regime: str,
                cell: int) -> spread.World:
    sizes = v2_contract["profiles"][profile]
    spec = v2_contract["world"]
    return spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(spec["degree"]), rewire=float(spec["rewire"]), seeds_per_item=int(spec["seeds_per_item"]),
        delay=int(spec["delay"]), visibility=float(spec["visibility"]), epsilon=float(spec["epsilon"]),
        refit_every=int(spec["refit_every"]), regime=regime, geometry=GEOMETRY,
        world_seed=int(contract["seeds"]["world_seed_offset"]) + cell,
    ))


def build_stream(world: spread.World, replicate: int, flip_rate: float) -> spread.Stream:
    """The prototype's stream, with the credibility sign reversed at rate ``flip_rate``.

    The sign draw is the prototype's own keyed uniform u: the sign is reversed
    when u >= 1 - flip_rate. At 0.5 that is the prototype's POS-sign rule.
    """

    config = world.config
    if config.random_credibility_sign or config.shift_credibility_weight is not None:
        raise ValueError("the flip rate replaces the prototype's sign and shift options")
    count = config.steps * config.items_per_step
    rng = np.random.default_rng(core.keyed_seed(config.world_seed, spread.TAG_ITEMS, replicate))
    features = rng.random((count, 4))
    sign_rng = np.random.default_rng(core.keyed_seed(config.world_seed, spread.TAG_SIGN, replicate))
    sign = -1 if sign_rng.random() >= 1.0 - float(flip_rate) else 1
    shape = spread._veracity_logit_without_intercept(config.geometry, features, 3.0 * sign)
    truth_probability = core.sigmoid(world.intercept + shape)
    veracity = rng.random(count) < truth_probability
    revealed = rng.random(count) < config.visibility
    seeds = np.stack([rng.choice(config.agents, config.seeds_per_item, replace=False) for _ in range(count)])
    born = np.repeat(np.arange(config.steps), config.items_per_step)
    return spread.Stream(features, veracity, revealed, seeds, born, None, sign)


def run_cell(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, regime: str, family: str,
             flip_rate: float, cell: int, device: str) -> dict[str, Any]:
    world = build_world(contract, v2_contract, profile, regime, cell)
    if float(flip_rate) not in [float(p) for p in contract["design"]["flip_rates"]]:
        raise ValueError(f"{flip_rate} is not a registered flip rate")

    def learner(replicate: int) -> spread.Policy:
        # The same seeds at every flip rate: the cell's world, item features, and
        # learner draws are shared across rates, so rates are paired.
        return v2._learner(v2_contract, world, family, 0, cell, replicate, device)

    donors = [spread.simulate(world, build_stream(world, r, flip_rate), learner(r), r).snapshots
              for r in v2_contract["seeds"]["donor_replicates"]]
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = build_stream(world, replicate, flip_rate)
        arms = {
            "F1": spread.prior_policy("F1", spread.hand_prior(world.config)),
            "T0_1": spread.table_policy("T0_1", donors[:1], final_only=False),
            "T0_4": spread.table_policy("T0_4", donors, final_only=False),
            "L0": learner(replicate),
        }
        rows.append({"replicate": replicate, "credibility_sign": stream.credibility_sign, "arms": {
            name: v2.metrics(measure.measured_simulate(world, stream, policy, replicate)[1])
            for name, policy in arms.items()
        }})
    return {"regime": regime, "family": family, "flip_rate": float(flip_rate), "cell": cell,
            "donor_signs": [build_stream(world, r, flip_rate).credibility_sign
                            for r in v2_contract["seeds"]["donor_replicates"]],
            "rows": rows}


# -- the registered rule -----------------------------------------------------------------


def _label(rule: Mapping[str, Any]) -> str:
    if not rule["material"]:
        return "INDISTINGUISHABLE"
    return "POSITIVE" if rule["mean_cell_delta"] > 0 else "NEGATIVE"


def group_key(regime: str, family: str, flip_rate: float) -> str:
    return f"{regime}/{family}/p={float(flip_rate):g}"


def summarize(contract: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["regime"], cell["family"], float(cell["flip_rate"])), []).append(cell)
    tag = int(contract["seeds"]["rule_tag"])
    table: dict[str, Any] = {}
    for g_index, ((regime, family, rate), members) in enumerate(sorted(groups.items())):
        members = sorted(members, key=lambda c: c["cell"])
        entry: dict[str, Any] = {"cells": len(members)}
        for m_index, name in enumerate(v2.MEASURES):
            rules = {}
            for c_index, (comparison, (left, right)) in enumerate(COMPARISONS.items()):
                means = [float(np.mean([r["arms"][left][name] - r["arms"][right][name] for r in c["rows"]]))
                         for c in members]
                if not np.all(np.isfinite(means)):
                    raise RuntimeError(f"a {name} is undefined in {regime}/{family}/{rate}")
                rules[comparison] = v2._group_rule(contract, means, (tag, g_index, m_index, c_index))
            entry[name] = rules
            entry[f"mean_{name}"] = {
                arm: round(float(np.mean([r["arms"][arm][name] for c in members for r in c["rows"]])), 4)
                for arm in ARMS
            }
        entry["flipped_run_share"] = round(float(np.mean([r["credibility_sign"] == -1 for c in members
                                                           for r in c["rows"]])), 4)
        table[group_key(regime, family, rate)] = entry
    primary = {key: entry[v2.PRIMARY] for key, entry in table.items()}
    return {"table": table, "dispositions": dispositions(contract, primary)}


def dispositions(contract: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    rates = [float(p) for p in contract["design"]["flip_rates"]]
    out: dict[str, Any] = {"onset": {}, "labels": {}}
    control_failures = []
    for regime, families in contract["design"]["families"].items():
        for family in families:
            present = [rate for rate in rates if group_key(regime, family, rate) in table]
            if not present:  # a smoke run may cover only some groups
                continue
            labels = {f"{rate:g}": {name: _label(table[group_key(regime, family, rate)][name]) for name in COMPARISONS}
                      for rate in present}
            out["labels"][f"{regime}/{family}"] = labels
            onset = next((rate for rate in present if labels[f"{rate:g}"]["best_table"] == "POSITIVE"), None)
            out["onset"][f"{regime}/{family}"] = "NONE_UP_TO_0.5" if onset is None else f"{onset:g}"
            if "0.5" in labels and labels["0.5"]["closed_loop"] != "POSITIVE":
                control_failures.append(f"{regime}/{family}")
    out["design_check"] = "POSITIVE_CONTROL_FAILS_AT_0.5" if control_failures else "POSITIVE_CONTROL_HOLDS_AT_0.5"
    out["control_failures"] = control_failures
    return out


# -- main --------------------------------------------------------------------------------


def _task(args: tuple[str, str, int, str, str, float, int]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    contract_path, profile, index, regime, family, rate, cell = args
    contract = load_contract(Path(contract_path))
    v2_contract = load_v2_contract(contract)
    v2.apply_learner_settings(v2_contract)
    device = v2.device_for(family)
    started = time.perf_counter()
    result = run_cell(contract, v2_contract, profile, regime, family, rate, cell, device)
    return index, result, {"device": device, "task_seconds": round(time.perf_counter() - started, 1)}


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
    v2_contract = load_v2_contract(contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    cells_per_group = args.pilot_cells or int(v2_contract["profiles"][args.profile]["cells"])
    specs = [(r, f, float(p), cell) for r, families in contract["design"]["families"].items() for f in families
             for p in contract["design"]["flip_rates"] for cell in range(cells_per_group)]
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
                    "group": group_key(cell["regime"], cell["family"], cell["flip_rate"]), "cell": cell["cell"],
                    **timing, "elapsed_seconds": round(time.perf_counter() - started, 1),
                }) + "\n")
    ordered = sorted(results.values(), key=lambda c: (c["regime"], c["family"], c["flip_rate"], c["cell"]))
    cells_path = args.output_dir / "cells.jsonl"
    cells_path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in ordered), encoding="utf-8")
    summary = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "cells_per_group": cells_per_group, "primary_measure": v2.PRIMARY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        **summarize(contract, ordered),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": v2._host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"onset": summary["dispositions"]["onset"],
                      "design_check": summary["dispositions"]["design_check"], **receipt["outputs"]}))


if __name__ == "__main__":
    main()
