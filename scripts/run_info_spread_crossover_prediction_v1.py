"""Information-spread crossover prediction v1 (recommended experiment 3, stage 1).

Design: docs/research/info-spread-crossover-prediction-v1.md
Contract: experiments/configs/info_spread_crossover_prediction_v1.yaml

The flip-rate study (O26) found that the adaptive learner L0 overtakes the
four-donor table T0-4 between flip rates 0.1 and 0.25. This study registers
a rule that predicts where a k-donor table is overtaken, from two quantities
measured before any comparison is run:

- between-run heterogeneity: how far apart learners end up in runs whose
  credibility relation has opposite signs;
- within-run evidence: how much a learner's policy varies across runs of one
  world, which falls as the run reveals more outcomes.

Both come from learner runs in worlds with one fixed sign, on worlds and
seeds that the comparison does not use. The plug-in rule compares the
expected squared error of a k-donor trajectory table with that of the
in-run learner, against the generator's true veracity probability on a fixed
probe set. The comparison is then run at visibilities, table sizes, and flip
rates that O26 did not run, and the rule's crossings are scored against the
observed ones and against two baselines.

A development run needs the precommit SHA; smoke and pilot runs skip that
check and are never read for results. cells.jsonl and summary.json carry no
timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import info_spread_unverified_measure_v1 as measure  # noqa: E402
import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_flip_rate_v1 as flip  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_crossover_prediction_v1.yaml"
SCHEMA = "info_spread_crossover_prediction.v1"
REGIME, FAMILY, GEOMETRY = "A", "linear_exact", "linear"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
TAG_PROBE = 40
NONE_CROSSING = 0.55
SCORED_TABLE_SIZES = (2, 4, 8)


class CrossoverError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossoverError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- contract ----------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == SCHEMA, "unexpected contract schema")
    require(contract.get("status") == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION", "not a precommit contract")
    for binding in contract["source_bindings"]:
        require(_sha256(ROOT / binding["path"]) == binding["sha256"], f"source binding drifted: {binding['path']}")
    design = contract["design"]
    require((design["regime"], design["family"], design["geometry"]) == (REGIME, FAMILY, GEOMETRY),
            "the study runs Regime A, the exact linear learner, and the linear geometry")
    rates = [float(p) for p in design["flip_rates"]]
    require(rates == sorted(rates) and rates[0] == 0.0 and rates[-1] == 0.5, "flip rates rise from 0 to 0.5")
    require(design["table_sizes"] == [1, 2, 4, 8], "the table sizes are 1, 2, 4, and 8")
    require(len(contract["seeds"]["donor_replicates"]) == max(design["table_sizes"]), "one donor per table slot")
    require(design["visibilities"] == [0.2, 0.5, 1.0], "the visibilities are 0.2, 0.5, and 1.0")
    require(contract["seeds"]["world_seed_offset"] != contract["measurement"]["world_seed_offset"],
            "the measurement worlds are not the comparison worlds")
    scoring = contract["scoring"]
    require(scoring["scored_table_sizes"] == list(SCORED_TABLE_SIZES), "crossings are scored for k = 2, 4, 8")
    require(float(scoring["none_crossing_value"]) == NONE_CROSSING, "no crossing scores as 0.55")
    require(float(scoring["accuracy_bound"]) == 0.05, "the accuracy bound is 0.05")
    require(contract["execution"]["thread_variables"] == list(THREAD_VARIABLES), "the thread variables differ")
    require(contract["boundary"]["empirical_rows_read"] == 0, "stage 1 reads no empirical row")


def load_v2_contract() -> dict[str, Any]:
    return v2.load_contract()


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    require(len(commit_sha) == 40 and all(c in "0123456789abcdef" for c in commit_sha), "a full lowercase SHA is needed")
    require(git("rev-parse", "HEAD") == commit_sha, "the precommit SHA is not HEAD")
    paths = list(contract["precommit"]["precommit_paths"])
    require(not git("status", "--porcelain", "--untracked-files=no", "--", *paths), "precommit paths are dirty")
    tracked = set(git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    require(tracked == set(paths), f"precommit paths missing from the commit: {sorted(set(paths) - tracked)}")
    return {"commit_sha": commit_sha, "precommit_paths": paths,
            "source_bindings": [{"path": b["path"], "sha256": b["sha256"]} for b in contract["source_bindings"]]}


# -- worlds, streams, learners -----------------------------------------------------------


def build_world(v2_contract: Mapping[str, Any], profile: str, cell: int, visibility: float, offset: int) -> spread.World:
    """The flip-rate study's world, with the visibility and seed offset of this study."""

    sizes = v2_contract["profiles"][profile]
    spec = v2_contract["world"]
    return spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(spec["degree"]), rewire=float(spec["rewire"]), seeds_per_item=int(spec["seeds_per_item"]),
        delay=int(spec["delay"]), visibility=float(visibility), epsilon=float(spec["epsilon"]),
        refit_every=int(spec["refit_every"]), regime=REGIME, geometry=GEOMETRY, world_seed=int(offset) + cell,
    ))


def learner(v2_contract: Mapping[str, Any], world: spread.World, cell: int, replicate: int) -> spread.Policy:
    return v2._learner(v2_contract, world, FAMILY, 0, cell, replicate, "cpu")


def probe(world: spread.World, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keyed uniform item features at the prototype's probe exposure, and the true probability under each sign."""

    rng = np.random.default_rng(core.keyed_seed(world.config.world_seed, TAG_PROBE))
    items = rng.random((size, 4))
    exposure = spread.probe_rows(world.config, 0.5)[0, 4:]
    rows = np.hstack([items, np.repeat(exposure[None, :], size, axis=0)])
    truth = {sign: core.sigmoid(world.intercept + spread._veracity_logit_without_intercept(GEOMETRY, items, 3.0 * sign))
             for sign in (1, -1)}
    return rows, truth[1], truth[-1]


def trajectory(snapshots: core.PolicySnapshots, rows: np.ndarray, steps: int) -> np.ndarray:
    """The policy in force at each step, evaluated on the probe rows: [steps, rows]."""

    cache: dict[int, np.ndarray] = {}
    out = np.empty((steps, len(rows)))
    for step in range(steps):
        position = int(np.searchsorted(np.asarray(snapshots.steps), step, side="right")) - 1
        if position not in cache:
            kind, value = snapshots.policies[position]
            require(kind in ("pooled", "prior"), "Regime A snapshots are pooled")
            cache[position] = value.predict(rows) if kind == "pooled" else value.probability(rows)
        out[step] = cache[position]
    return out


# -- measurement: heterogeneity and evidence, from runs with one fixed sign -------------


def measure_cell(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, visibility: float,
                 cell: int) -> dict[str, Any]:
    """Per-step components of the plug-in rule for one measurement world."""

    spec = contract["measurement"]
    world = build_world(v2_contract, profile, cell, visibility, int(spec["world_seed_offset"]))
    rows, f_plus, f_minus = probe(world, int(spec["probe_size"]))
    steps = world.config.steps
    paths: dict[int, np.ndarray] = {}
    for sign, rate in ((1, 0.0), (-1, 1.0)):
        runs = []
        for replicate in range(int(spec["replicates"])):
            stream = flip.build_stream(world, replicate, rate)
            require(stream.credibility_sign == sign, "a measurement stream has the wrong sign")
            result = spread.simulate(world, stream, learner(v2_contract, world, cell, replicate), replicate)
            runs.append(trajectory(result.snapshots, rows, steps))
        paths[sign] = np.stack(runs)  # [replicates, steps, rows]
    mean_plus, mean_minus = paths[1].mean(axis=0), paths[-1].mean(axis=0)
    d = mean_minus - mean_plus
    components = {
        "A_plus": np.mean((mean_plus - f_plus) ** 2, axis=1),
        "A_minus": np.mean((mean_plus - f_minus) ** 2, axis=1),
        "C_plus": np.mean((mean_plus - f_plus) * d, axis=1),
        "C_minus": np.mean((mean_plus - f_minus) * d, axis=1),
        "Dd": np.mean(d ** 2, axis=1),
        "V_plus": np.mean(paths[1].var(axis=0, ddof=1), axis=1),
        "V_minus": np.mean(paths[-1].var(axis=0, ddof=1), axis=1),
    }
    return {"visibility": float(visibility), "cell": cell,
            "components": {key: [float(v) for v in value] for key, value in components.items()}}


def plug_in_gap(components: Mapping[str, Sequence[float]], k: int, p: float) -> float:
    """Expected squared error of a k-donor trajectory table minus the in-run learner's, averaged over steps.

    Donor signs are binomial; donor and run errors are independent given their
    signs. Positive means the learner is predicted to beat the table.
    """

    c = {key: np.asarray(value, dtype=np.float64) for key, value in components.items()}
    learner_mse = (1 - p) * (c["A_plus"] + c["V_plus"]) + p * (c["A_minus"] + 2 * c["C_minus"] + c["Dd"] + c["V_minus"])
    table_mse = np.zeros_like(learner_mse)
    for run_sign, weight in (("plus", 1 - p), ("minus", p)):
        if weight == 0:
            continue
        for j in range(k + 1):
            chance = comb(k, j) * p**j * (1 - p) ** (k - j)
            if chance == 0:
                continue
            share = j / k
            error = (c[f"A_{run_sign}"] + 2 * share * c[f"C_{run_sign}"] + share**2 * c["Dd"]
                     + ((k - j) * c["V_plus"] + j * c["V_minus"]) / k**2)
            table_mse += weight * chance * error
    return float(np.mean(table_mse - learner_mse))


def simple_index(components: Mapping[str, Sequence[float]], p: float) -> float:
    """Between-run heterogeneity over within-run variance, averaged over steps: p(1 - p) Dd / V."""

    dd = float(np.mean(components["Dd"]))
    v = float(np.mean((np.asarray(components["V_plus"]) + np.asarray(components["V_minus"])) / 2))
    return p * (1 - p) * dd / v


def _sustained_crossing(rates: Sequence[float], values: Sequence[float]) -> float | None:
    """Where the curve last rises above zero to stay there; None if it ends at or below zero."""

    if values[-1] <= 0:
        return None
    last = max((i for i, value in enumerate(values) if value <= 0), default=None)
    if last is None:
        return float(rates[0])
    p0, p1, v0, v1 = rates[last], rates[last + 1], values[last], values[last + 1]
    return float(p0 + (p1 - p0) * (0 - v0) / (v1 - v0))


def predicted_crossings(contract: Mapping[str, Any], measured: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grid = np.round(np.arange(0, 0.5 + 1e-9, float(contract["scoring"]["prediction_grid_step"])), 6)
    out: dict[str, Any] = {}
    for visibility in contract["design"]["visibilities"]:
        cells = [m["components"] for m in measured if m["visibility"] == float(visibility)]
        require(len(cells) == int(contract["measurement"]["cells"]), "a measurement cell is missing")
        for k in contract["design"]["table_sizes"]:
            gaps = [float(np.mean([plug_in_gap(c, k, float(p)) for c in cells])) for p in grid]
            crossing = _sustained_crossing(list(grid), gaps)
            out[f"pi={visibility:g}/k={k}"] = {
                "crossing": None if crossing is None else round(crossing, 4),
                "gap_at_registered_rates": {f"{p:g}": round(float(np.mean([plug_in_gap(c, k, float(p)) for c in cells])), 8)
                                            for p in contract["design"]["flip_rates"]},
                "simple_index_threshold": round((k - 1) / (k + 1), 6),
            }
        out[f"pi={visibility:g}/components"] = {
            "mean_Dd": round(float(np.mean([np.mean(c["Dd"]) for c in cells])), 8),
            "mean_V": round(float(np.mean([np.mean((np.asarray(c["V_plus"]) + np.asarray(c["V_minus"])) / 2)
                                           for c in cells])), 8),
            "simple_index_at_rates": {f"{p:g}": round(float(np.mean([simple_index(c, float(p)) for c in cells])), 6)
                                      for p in contract["design"]["flip_rates"]},
        }
    return out


def baseline_crossing(name: str, k: int, contract: Mapping[str, Any]) -> float:
    spec = contract["scoring"]["baselines"][name]
    if name == "constant":
        return float(spec["crossing"])
    # The k-only rule: p(1 - p) = c (k - 1) / (k + 1), with c set by the O26 midpoint at k = 4.
    c = float(spec["c"])
    target = c * (k - 1) / (k + 1)
    return float((1 - np.sqrt(1 - 4 * target)) / 2)


# -- the comparison ----------------------------------------------------------------------


def arms_for(donors: Sequence[core.PolicySnapshots], policy: spread.Policy, sizes: Sequence[int]) -> dict[str, spread.Policy]:
    arms = {"L0": policy}
    for k in sizes:
        arms[f"T0_{k}"] = spread.table_policy(f"T0_{k}", donors[:k], final_only=False)
    return arms


def run_cell(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, visibility: float,
             flip_rate: float, cell: int) -> dict[str, Any]:
    world = build_world(v2_contract, profile, cell, visibility, int(contract["seeds"]["world_seed_offset"]))
    donor_replicates = [int(r) for r in contract["seeds"]["donor_replicates"]]
    donors = [spread.simulate(world, flip.build_stream(world, r, flip_rate), learner(v2_contract, world, cell, r), r).snapshots
              for r in donor_replicates]
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = flip.build_stream(world, replicate, flip_rate)
        arms = arms_for(donors, learner(v2_contract, world, cell, replicate), contract["design"]["table_sizes"])
        rows.append({"replicate": replicate, "credibility_sign": stream.credibility_sign, "arms": {
            name: v2.metrics(measure.measured_simulate(world, stream, policy, replicate)[1]) for name, policy in arms.items()
        }})
    return {"visibility": float(visibility), "flip_rate": float(flip_rate), "cell": cell,
            "donor_signs": [flip.build_stream(world, r, flip_rate).credibility_sign for r in donor_replicates],
            "rows": rows}


def group_key(visibility: float, k: int, flip_rate: float) -> str:
    return f"pi={float(visibility):g}/k={k}/p={float(flip_rate):g}"


def observed(contract: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The group rule and mean cell delta of L0 - T0_k for every visibility, table size, and flip rate."""

    tag = int(contract["seeds"]["rule_tag"])
    design = contract["design"]
    table: dict[str, Any] = {}
    for v_index, visibility in enumerate(design["visibilities"]):
        for r_index, rate in enumerate(design["flip_rates"]):
            members = sorted([c for c in cells if c["visibility"] == float(visibility) and c["flip_rate"] == float(rate)],
                             key=lambda c: c["cell"])
            if not members:
                continue
            for k_index, k in enumerate(design["table_sizes"]):
                means = [float(np.mean([r["arms"]["L0"][v2.PRIMARY] - r["arms"][f"T0_{k}"][v2.PRIMARY] for r in c["rows"]]))
                         for c in members]
                require(bool(np.all(np.isfinite(means))), f"an undefined measure in {group_key(visibility, k, rate)}")
                rule = v2._group_rule(contract, means, (tag, v_index, r_index, k_index))
                label = "INDISTINGUISHABLE" if not rule["material"] else ("POSITIVE" if rule["mean_cell_delta"] > 0 else "NEGATIVE")
                table[group_key(visibility, k, rate)] = {**rule, "label": label, "cells": len(members),
                                                         "exact_mean_cell_delta": float(np.mean(means))}
    return table


def score(contract: Mapping[str, Any], predicted: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    design, scoring = contract["design"], contract["scoring"]
    rates = [float(p) for p in design["flip_rates"]]
    series: dict[str, Any] = {}
    for visibility in design["visibilities"]:
        for k in design["table_sizes"]:
            values = [table[group_key(visibility, k, p)]["exact_mean_cell_delta"] for p in rates]
            crossing = _sustained_crossing(rates, values)
            series[f"pi={visibility:g}/k={k}"] = {
                "observed": None if crossing is None else round(crossing, 4),
                "predicted": predicted[f"pi={visibility:g}/k={k}"]["crossing"],
                **{f"baseline_{name}": round(baseline_crossing(name, k, contract), 4) for name in scoring["baselines"]},
            }

    def value(x: float | None) -> float:
        return NONE_CROSSING if x is None else float(x)

    scored = {key: row for key, row in series.items() if int(key.split("k=")[1]) in SCORED_TABLE_SIZES}
    errors = {name: [abs(value(row[name]) - value(row["observed"])) for row in scored.values()]
              for name in ("predicted", *(f"baseline_{b}" for b in scoring["baselines"]))}
    bound = float(scoring["accuracy_bound"])
    mean_error = {name: round(float(np.mean(errs)), 4) for name, errs in errors.items()}
    within = {name: int(sum(e <= bound for e in errs)) for name, errs in errors.items()}
    rule_error = mean_error["predicted"]
    best_baseline = min(mean_error[f"baseline_{b}"] for b in scoring["baselines"])
    if rule_error <= bound and rule_error < best_baseline:
        disposition = "THE_RULE_PREDICTS_THE_CROSSOVER"
    elif rule_error < best_baseline:
        disposition = "THE_RULE_BEATS_THE_BASELINES_BUT_MISSES_BY_MORE_THAN_THE_BOUND"
    else:
        disposition = "THE_RULE_DOES_NOT_BEAT_THE_BASELINES"
    return {"series": series, "scored_series": sorted(scored), "mean_absolute_error": mean_error,
            "within_bound": within, "status": f"E3_{disposition}", "disposition": disposition}


def expectations(contract: Mapping[str, Any], table: Mapping[str, Any], scored: Mapping[str, Any]) -> dict[str, Any]:
    design = contract["design"]
    visibilities = design["visibilities"]
    zero = {f"pi={v:g}/k={k}": table[group_key(v, k, 0.0)]["exact_mean_cell_delta"] for v in visibilities for k in (2, 4, 8)}
    x1 = all(value <= 0 for value in zero.values())
    x5 = {f"pi={v:g}": table[group_key(v, 1, 0.0)]["label"] for v in visibilities}

    def crossing(v: float, k: int) -> float:
        value = scored["series"][f"pi={v:g}/k={k}"]["observed"]
        return NONE_CROSSING if value is None else value

    x2 = all(crossing(v, 2) <= crossing(v, 4) <= crossing(v, 8) for v in visibilities)
    x3 = all(crossing(0.2, k) >= crossing(0.5, k) >= crossing(1.0, k) for k in (2, 4, 8))
    return {
        "X1": {"held": x1, "mean_cell_delta_at_p_0": {k: round(v, 4) for k, v in zero.items()}},
        "X2": {"held": x2},
        "X3": {"held": x3},
        "X4": {"held": scored["disposition"] == "THE_RULE_PREDICTS_THE_CROSSOVER"},
        "X5": {"held": all(label == "INDISTINGUISHABLE" for label in x5.values()), "labels": x5},
    }


# -- run ---------------------------------------------------------------------------------


def _measure_task(args: tuple[str, str, int, float, int]) -> tuple[str, int, dict[str, Any], float]:
    contract_path, profile, index, visibility, cell = args
    contract = load_contract(Path(contract_path))
    v2_contract = load_v2_contract()
    v2.apply_learner_settings(v2_contract)
    started = time.perf_counter()
    result = measure_cell(contract, v2_contract, profile, visibility, cell)
    return "measure", index, result, round(time.perf_counter() - started, 1)


def _compare_task(args: tuple[str, str, int, float, float, int]) -> tuple[str, int, dict[str, Any], float]:
    contract_path, profile, index, visibility, rate, cell = args
    contract = load_contract(Path(contract_path))
    v2_contract = load_v2_contract()
    v2.apply_learner_settings(v2_contract)
    started = time.perf_counter()
    result = run_cell(contract, v2_contract, profile, visibility, rate, cell)
    return "compare", index, result, round(time.perf_counter() - started, 1)


def _task(args: tuple) -> tuple[str, int, dict[str, Any], float]:
    return _measure_task(args[1:]) if args[0] == "measure" else _compare_task(args[1:])


def plan(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, measure_cells: int | None,
         compare_cells: int | None) -> list[tuple]:
    design = contract["design"]
    m_cells = measure_cells if measure_cells is not None else int(contract["measurement"]["cells"])
    c_cells = compare_cells if compare_cells is not None else int(v2_contract["profiles"][profile]["cells"])
    tasks: list[tuple] = []
    for visibility in design["visibilities"]:
        for cell in range(m_cells):
            tasks.append(("measure", len(tasks), float(visibility), cell))
    for visibility in design["visibilities"]:
        for rate in design["flip_rates"]:
            for cell in range(c_cells):
                tasks.append(("compare", len(tasks), float(visibility), float(rate), cell))
    # Highest visibility first: more revealed rows make the slowest refits.
    tasks.sort(key=lambda t: (-t[2], t[1]))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", choices=("development", "smoke"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--pilot-cells", type=int, default=None,
                        help="a full-load pilot with this many cells per group and measurement visibility; never read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    v2_contract = load_v2_contract()
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    if is_registered_run:
        require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), f"set {', '.join(THREAD_VARIABLES)} to 1")
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    if args.profile == "smoke":
        tasks = plan(contract, v2_contract, "smoke", 2, 1)
    else:
        tasks = plan(contract, v2_contract, args.profile, args.pilot_cells, args.pilot_cells)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
    measured: list[dict[str, Any]] = []
    compared: list[dict[str, Any]] = []
    payload = [(t[0], str(args.contract), args.profile, *t[1:]) for t in tasks]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (kind, index, result, seconds) in enumerate(pool.map(_task, payload, chunksize=1), start=1):
            (measured if kind == "measure" else compared).append(result)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"done": done, "of": len(tasks), "kind": kind, "task": index,
                                         "task_seconds": seconds,
                                         "elapsed_seconds": round(time.perf_counter() - started, 1)}) + "\n")
    measured.sort(key=lambda m: (m["visibility"], m["cell"]))
    compared.sort(key=lambda c: (c["visibility"], c["flip_rate"], c["cell"]))
    (args.output_dir / "measurement.jsonl").write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in measured),
                                                      encoding="utf-8")
    (args.output_dir / "cells.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in compared),
                                                encoding="utf-8")
    summary: dict[str, Any] = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "primary_measure": v2.PRIMARY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
    }
    if args.profile == "development" and args.pilot_cells is None:
        predicted = predicted_crossings(contract, measured)
        table = observed(contract, compared)
        scored = score(contract, predicted, table)
        summary.update({"predicted": predicted, "observed": table, "scoring": scored,
                        "expectations": expectations(contract, table, scored), "result_status": scored["status"]})
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    outputs = {name: _sha256(args.output_dir / name) for name in ("measurement.jsonl", "cells.jsonl", "summary.json")}
    receipt = {"precommit": precommit, "host": v2._host(), "workers": args.workers,
               "wall_seconds": round(time.perf_counter() - started, 1), "outputs": outputs}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary.get("result_status", summary["status"]), **outputs,
                      "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    main()
