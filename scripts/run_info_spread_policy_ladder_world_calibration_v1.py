"""Information-spread policy ladder, world-calibration study v1 (O33).

Design: docs/research/info-spread-policy-ladder-world-calibration-v1.md
Contract: experiments/configs/info_spread_policy_ladder_world_calibration_v1.yaml

O28 split each LLM prior into its direction (the model's answers) and its
calibration (location and steepness over D3b's design cohort), and found that
the utility cost of D5's prior inside the learner comes from its calibration
(C63), while the unclipped prior's two parts interact (C64). But the
recalibration was matched on the cohort, and in the world it overshot (C66):
at F1's calibration an LLM direction shares 1.35 to 1.48 times as much as F1.

This study repeats O28's 2 x 2 with the location matched in the world. Each
recalibrated prior keeps O28's weights (its direction at its calibration
source's steepness over the cohort). Its intercept is then set on separate
calibration worlds, so that the fixed prior shares as many unverified items
per run as its calibration source does there:

- if O28's cohort-matched intercept already shares within the contract's
  keep tolerance of the target, it is kept;
- otherwise the intercept is the largest one whose share count does not
  exceed the target, found by bisection.

The calibration worlds come from study v2's generator at their own seeds,
used by no registered study. A world-match check then asks, out of sample on
the study cells, whether each recalibrated fixed prior shares within the
contract's tolerance of its source. The learners, measure, recorded inputs,
parity, and consistency checks are O28's. A development run needs the
precommit SHA; smoke and pilot runs skip that check and are never read for
results. cells.jsonl and summary.json carry no timing.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS needs this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_prior_calibration_v1 as o28  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_unclipped_prior_v1 as unclipped  # noqa: E402
import run_info_spread_policy_ladder_utility_v1 as o25  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_world_calibration_v1.yaml"
# Each world-calibrated prior: the O28 prior whose weights it keeps.
PRIORS = {"P0_hcalw": "P0_hcal", "F1_pcalw": "F1_pcal", "P0u_hcalw": "P0u_hcal", "F1_ucalw": "F1_ucal"}
LEARNERS = {"L1_hcalw": "P0_hcalw", "L0_pcalw": "F1_pcalw", "L1u_hcalw": "P0u_hcalw", "L0_ucalw": "F1_ucalw"}
RECORDED_LEARNERS = o28.RECORDED_LEARNERS
COMPARISONS = {
    "P0": {
        "calibration_given_llm_answers": ("L1", "L1_hcalw"),
        "calibration_given_hand_answers": ("L0_pcalw", "L0"),
        "answers_given_hand_calibration": ("L1_hcalw", "L0"),
        "answers_given_llm_calibration": ("L1", "L0_pcalw"),
    },
    "P0u": {
        "calibration_given_llm_answers": ("L1u", "L1u_hcalw"),
        "calibration_given_hand_answers": ("L0_ucalw", "L0"),
        "answers_given_hand_calibration": ("L1u_hcalw", "L0"),
        "answers_given_llm_calibration": ("L1u", "L0_ucalw"),
    },
}
TOTALS = o28.TOTALS
CALIBRATION = o28.CALIBRATION
ANSWERS = o28.ANSWERS
LABELS = o28.LABELS
UTILITY = o28.UTILITY


def flat_comparisons() -> dict[str, tuple[str, str]]:
    return {f"{prior}/{name}": pair for prior, named in COMPARISONS.items() for name, pair in named.items()}


def sources(name: str) -> tuple[str, str]:
    """(direction source, calibration source) of a world-calibrated prior."""

    return o28.PRIORS[PRIORS[name]]


# -- contract --------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    if contract["priors"] != PRIORS or contract["learner_arms"] != LEARNERS:
        raise ValueError("arms differ from the runner's")
    if {k: tuple(v) for k, v in contract["comparisons"].items()} != flat_comparisons():
        raise ValueError("comparisons differ from the runner's")
    return contract


def contracts(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return o28.contracts(contract)


verify_source_bindings = v2.verify_source_bindings
verify_precommit = v2.verify_precommit
_sha256 = v2._sha256


# -- calibration worlds ------------------------------------------------------------------------


def calibration_runs(contract: Mapping[str, Any], v2_contract: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """(geometry, world index, replicate) of every calibration run, in a fixed order."""

    spec = contract["world_calibration"]
    return [(geometry, index, replicate) for geometry in v2_contract["design"]["study_geometries"]
            for index in range(int(spec["worlds_per_geometry"])) for replicate in spec["replicates"]]


def calibration_world(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, regime: str,
                      geometry: str, index: int) -> spread.World:
    """Study v2's world generator at the calibration seeds."""

    spec = contract["world_calibration"]
    geometries = list(v2_contract["design"]["study_geometries"])
    shifted = {**v2_contract, "seeds": {**v2_contract["seeds"], "world_seed_offset": int(spec["world_seed_offset"])}}
    cell = geometries.index(geometry) * int(spec["worlds_per_geometry"]) + index
    return v2.build_world(shifted, profile, regime, geometry, cell)


def calibration_shares(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], profile: str, regime: str,
                       geometry: str, index: int, replicate: int, prior: core.LinearPrior) -> int:
    world = calibration_world(contract, v2_contract, profile, regime, geometry, index)
    stream = spread.build_stream(world, replicate)
    return o28._shares(o25._measured(world, stream, spread.prior_policy("calibration", prior), replicate))


def _calibration_task(args: tuple[str, str, str, str, int, int, tuple[float, ...], float]) -> int:
    contract_path, profile, regime, geometry, index, replicate, weight, bias = args
    contract = load_contract(Path(contract_path))
    v2_contract, _llm, _o24 = contracts(contract)
    prior = core.LinearPrior(np.asarray(weight, dtype=np.float64), bias)
    return calibration_shares(contract, v2_contract, profile, regime, geometry, index, replicate, prior)


def mean_shares(pool: Executor, contract_path: str, contract: Mapping[str, Any], v2_contract: Mapping[str, Any],
                profile: str, regime: str, prior: core.LinearPrior) -> float:
    runs = calibration_runs(contract, v2_contract)
    weight = tuple(float(w) for w in prior.weight)
    jobs = [(contract_path, profile, regime, g, i, r, weight, float(prior.bias)) for g, i, r in runs]
    return float(np.mean(list(pool.map(_calibration_task, jobs))))


def locate(shares_of: Callable[[float], float], target: float, cohort_bias: float,
           spec: Mapping[str, Any]) -> dict[str, Any]:
    """The intercept whose prior shares like the target on the calibration worlds."""

    cohort = shares_of(cohort_bias)
    if abs(cohort / target - 1.0) <= float(spec["keep_tolerance"]):
        return {"method": "KEPT_COHORT_LOCATION", "bias": cohort_bias, "target": target, "achieved": cohort}
    low, high = (float(v) for v in spec["bisection_interval"])
    at_low, at_high = shares_of(low), shares_of(high)
    if not at_low <= target < at_high:
        raise RuntimeError("the target share count is outside the bisection interval")
    for _ in range(int(spec["bisection_steps"])):
        middle = 0.5 * (low + high)
        value = shares_of(middle)
        if value <= target:
            low, at_low = middle, value
        else:
            high, at_high = middle, value
    return {"method": "BISECTED", "bias": low, "target": target, "achieved": at_low, "just_above": at_high,
            "cohort_location_shares": cohort}


def world_calibrate(pool: Executor, contract_path: str, contract: Mapping[str, Any], v2_contract: Mapping[str, Any],
                    cohort_priors: Mapping[str, Mapping[str, core.LinearPrior]], profile: str) -> dict[str, Any]:
    """The world-located intercept of every recalibrated prior, per regime, with a report."""

    spec = contract["world_calibration"]
    biases: dict[str, dict[str, float]] = {}
    report: dict[str, Any] = {}
    for regime in v2_contract["design"]["regimes"]:
        priors = cohort_priors[regime]
        targets = {name: mean_shares(pool, contract_path, contract, v2_contract, profile, regime, priors[name])
                   for name in RECORDED_LEARNERS}
        biases[regime], report[regime] = {}, {"source_shares": {k: round(v, 3) for k, v in targets.items()}}
        for name, cohort_name in PRIORS.items():
            _direction, source = sources(name)
            weight = priors[cohort_name].weight

            def shares_of(bias: float, weight: np.ndarray = weight) -> float:
                return mean_shares(pool, contract_path, contract, v2_contract, profile, regime,
                                   core.LinearPrior(weight, bias))

            located = locate(shares_of, targets[source], float(priors[cohort_name].bias), spec)
            biases[regime][name] = located["bias"]
            report[regime][name] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in located.items()}
    return {"biases": biases, "report": report}


def world_priors(cohort_priors: Mapping[str, Mapping[str, core.LinearPrior]],
                 biases: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, core.LinearPrior]]:
    out = {}
    for regime, by_name in biases.items():
        out[regime] = {name: core.LinearPrior(cohort_priors[regime][PRIORS[name]].weight, bias)
                       for name, bias in by_name.items()}
    return out


# -- cells -------------------------------------------------------------------------------


def run_fixed_cell(v2_contract: Mapping[str, Any], priors: Mapping[str, Any], profile: str, regime: str,
                   geometry: str, cell: int) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        rows.append({"replicate": replicate, "arms": {
            name: o25._measured(world, stream, spread.prior_policy(name, priors[regime][name]), replicate)
            for name in PRIORS
        }})
    return {"kind": "fixed", "regime": regime, "geometry": geometry, "cell": cell, "rows": rows}


def run_learner_cell(v2_contract: Mapping[str, Any], priors: Mapping[str, Any], profile: str, regime: str,
                     family: str, geometry: str, cell: int, device: str) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    gi = list(v2_contract["design"]["geometries"]).index(geometry)
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        rows.append({"replicate": replicate, "arms": {
            name: o25._measured(world, stream, llm._learner(v2_contract, world, family, gi, cell, replicate, device,
                                                            priors[regime][prior]), replicate)
            for name, prior in LEARNERS.items()
        }})
    stream = spread.build_stream(world, 0)
    parity = o25._measured(world, stream, llm._learner(v2_contract, world, family, gi, cell, 0, device,
                                                       spread.hand_prior(world.config)), 0)
    return {"kind": "learner", "regime": regime, "family": family, "geometry": geometry, "cell": cell,
            "rows": rows, "l0_replicate_0": parity}


# -- the registered rule -----------------------------------------------------------------


def _merged(recorded: Mapping[tuple, list], cells: Sequence[Mapping[str, Any]]) -> dict[tuple, list]:
    learners = {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in cells if c["kind"] == "learner"}
    out = {}
    for key, learner in sorted(learners.items()):
        rows = []
        for old, new in zip(recorded[key], learner["rows"], strict=True):
            if old["replicate"] != new["replicate"]:
                raise RuntimeError("replicates are not aligned")
            arms = {arm: m[UTILITY] for arm, m in old["arms"].items() if arm in RECORDED_LEARNERS.values()}
            arms.update({arm: m[UTILITY] for arm, m in new["arms"].items()})
            rows.append(arms)
        out[key] = rows
    return out


def world_match_check(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], recorded: Mapping[tuple, list],
                      cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Out of sample: does each world-calibrated fixed prior share like its source on the study cells?"""

    design = v2_contract["design"]
    tolerance = float(contract["dispositions"]["world_match_tolerance"])
    fixed = [c for c in cells if c["kind"] == "fixed" and c["geometry"] in design["study_geometries"]]
    result = {}
    for regime in design["regimes"]:
        first_family = design["families"][regime][0]
        shares: dict[str, list[int]] = {arm: [] for arm in (*PRIORS, *RECORDED_LEARNERS)}
        for c in (c for c in fixed if c["regime"] == regime):
            old = recorded[(regime, first_family, c["geometry"], c["cell"])]
            for old_row, new_row in zip(old, c["rows"], strict=True):
                for arm in RECORDED_LEARNERS:
                    shares[arm].append(old_row["arms"][arm]["unverified_shares"])
                for arm in PRIORS:
                    shares[arm].append(o28._shares(new_row["arms"][arm]))
        means = {arm: float(np.mean(v)) for arm, v in shares.items()}
        for name in PRIORS:
            direction, source = sources(name)
            ratio = means[name] / means[source]
            result[f"{name}/{regime}"] = {
                "mean_unverified_shares": round(means[name], 1), "calibration_source": round(means[source], 1),
                "direction_source": round(means[direction], 1), "ratio_to_calibration_source": round(ratio, 4),
                "matches": abs(ratio - 1.0) <= tolerance,
            }
    return result


def summarize(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], records: Mapping[str, Any],
              recorded: Mapping[tuple, list], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    o25_record = records["utility"]
    merged = _merged(recorded, cells)
    groups: dict[tuple[str, str, str], list[list[dict[str, float]]]] = {}
    for (regime, family, geometry, _cell), rows in sorted(merged.items()):
        groups.setdefault((regime, family, geometry), []).append(rows)
    minimum = {k: float(v) for k, v in o25_record["minimum_effect"].items()}
    tag, resamples = int(contract["seeds"]["rule_tag"]), int(contract["rule"]["bootstrap_resamples"])
    comparisons = flat_comparisons()
    arms = (*RECORDED_LEARNERS.values(), *LEARNERS)
    table: dict[str, Any] = {}
    consistency_failures = []
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        group = f"{regime}/{family}/{geometry}"
        entry: dict[str, Any] = {"cells": len(members)}
        for c_index, (name, (left, right)) in enumerate(comparisons.items()):
            means = [float(np.mean([r[left] - r[right] for r in rows])) for rows in members]
            entry[name] = o25.group_rule(means, (tag, g_index, c_index), minimum[regime], resamples)
        for prior, (left, right, o25_name) in TOTALS.items():
            total = round(float(np.mean([float(np.mean([r[left] - r[right] for r in rows])) for rows in members])), 4)
            entry[f"{prior}/total"] = total
            if total != o25_record["table"][group][o25_name]["mean_cell_delta"]:
                consistency_failures.append(f"{group}/{prior}")
        entry["mean_utility"] = {arm: round(float(np.mean([r[arm] for rows in members for r in rows])), 3)
                                 for arm in arms}
        table[group] = entry
    check = world_match_check(contract, v2_contract, recorded, cells)
    return {"minimum_effect": o25_record["minimum_effect"], "table": table,
            "learner_description": o28.learner_description(recorded, cells), "world_match_check": check,
            "consistency_failures": consistency_failures, "parity_failures": o28.parity_failures(recorded, cells),
            "dispositions": dispositions(contract, v2_contract, records, table, check)}


def dispositions(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], records: Mapping[str, Any],
                 table: Mapping[str, Any], check: Mapping[str, Any]) -> dict[str, Any]:
    design = v2_contract["design"]
    at_least = int(contract["dispositions"]["mostly_at_least"])
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    o28_cases = records["prior_calibration"]["dispositions"]["by_case"]
    by_case: dict[str, Any] = {}
    for prior, named in COMPARISONS.items():
        for regime in design["regimes"]:
            counts = {}
            for name in named:
                labels = [o25._label(v[f"{prior}/{name}"]) for k, v in study.items() if k.startswith(f"{regime}/")]
                counts[name] = {label: labels.count(label) for label in LABELS}
            summary = {name: o28._summary_label(c, at_least) for name, c in counts.items()}
            calibration = all(summary[n] == "MOSTLY_NEGATIVE" for n in CALIBRATION)
            answers = all(summary[n] == "MOSTLY_NEGATIVE" for n in ANSWERS)
            matched = all(check[f"{name}/{regime}"]["matches"] for name in PRIORS if prior in sources(name))
            attribution = ("WORLD_MATCH_FAILED" if not matched else
                           {(True, False): "CALIBRATION", (False, True): "ANSWERS", (True, True): "BOTH",
                            (False, False): "NEITHER_CONSISTENTLY"}[(calibration, answers)])
            case = f"{prior}/{regime}"
            by_case[case] = {"by_comparison": summary, "counts": counts, "attribution": attribution,
                             "o28_attribution": o28_cases[case]["attribution"]}
    by_o25 = records["utility"]["dispositions"]["by_comparison"]
    costed = [f"{prior}/{regime}" for prior, (_l, _r, name) in TOTALS.items() for regime in design["regimes"]
              if by_o25[f"{name}/{regime}"]["utility"] == "MOSTLY_NEGATIVE"]
    if costed != list(contract["dispositions"]["costed_cases"]):
        raise RuntimeError(f"O25's costed cases {costed} differ from the contract's")
    if any(by_case[case]["attribution"] == "WORLD_MATCH_FAILED" for case in costed):
        overall = "O33_WORLD_MATCH_FAILED"
        holds = {"C63": None, "C64": None}
    else:
        holds = {"C63": all(by_case[c]["attribution"] == "CALIBRATION" for c in ("P0/A", "P0/B")),
                 "C64": by_case["P0u/B"]["attribution"] == "NEITHER_CONSISTENTLY"}
        overall = {(True, True): "O33_C63_AND_C64_HOLD", (True, False): "O33_ONLY_C63_HOLDS",
                   (False, True): "O33_ONLY_C64_HOLDS", (False, False): "O33_NEITHER_HOLDS"}[(holds["C63"], holds["C64"])]
    return {"by_case": by_case, "costed_cases": costed, "holds": holds, "overall": overall}


# -- main --------------------------------------------------------------------------------


def _context(contract_path: str, elicitation_dir: str, biases: Mapping[str, Mapping[str, float]]) -> tuple[dict, dict]:
    contract = load_contract(Path(contract_path))
    v2_contract, llm_contract, o24_contract = contracts(contract)
    log_odds = unclipped.load_registered_elicitation(o24_contract, llm_contract, Path(elicitation_dir))
    return v2_contract, world_priors(o28.build_priors(llm_contract, log_odds)["priors"], biases)


def _learner_task(args: tuple) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, biases, profile, regime, family, geometry, cell = args
    v2_contract, priors = _context(contract_path, elicitation_dir, biases)
    v2.apply_learner_settings(v2_contract)
    device = v2.device_for(family)
    started = time.perf_counter()
    result = run_learner_cell(v2_contract, priors, profile, regime, family, geometry, cell, device)
    return (f"learner/{regime}/{family}/{geometry}/{cell}", result,
            {"device": device, "task_seconds": round(time.perf_counter() - started, 1)})


def _fixed_task(args: tuple) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, biases, profile, regime, geometry, cell = args
    v2_contract, priors = _context(contract_path, elicitation_dir, biases)
    started = time.perf_counter()
    result = run_fixed_cell(v2_contract, priors, profile, regime, geometry, cell)
    return f"fixed/{regime}/{geometry}/{cell}", result, {"device": "cpu",
                                                          "task_seconds": round(time.perf_counter() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", choices=("development", "smoke"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--pilot-cells", type=int, default=None,
                        help="a full-load pilot with this many cells per group; never read for results")
    parser.add_argument("--elicitation-dir", type=Path, required=True, help="D5's first registered elicitation")
    parser.add_argument("--v2-cells", type=Path)
    parser.add_argument("--d5-cells", type=Path)
    parser.add_argument("--o24-cells", type=Path)
    parser.add_argument("--o25-cells", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    v2_contract, llm_contract, o24_contract = contracts(contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    inputs = {"study_v2": args.v2_cells, "llm_arms": args.d5_cells, "unclipped_prior": args.o24_cells,
              "utility": args.o25_cells}
    if is_registered_run and any(p is None for p in inputs.values()):
        parser.error("a registered run needs --v2-cells, --d5-cells, --o24-cells, and --o25-cells")
    log_odds = unclipped.load_registered_elicitation(o24_contract, llm_contract, args.elicitation_dir)
    built = o28.build_priors(llm_contract, log_odds)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        calibrated = world_calibrate(pool, str(args.contract), contract, v2_contract, built["priors"], args.profile)
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"task": "world_calibration", "task_seconds": round(time.perf_counter() - started, 1),
                                 "elapsed_seconds": round(time.perf_counter() - started, 1)}) + "\n")
    design = v2_contract["design"]
    cells_per_group = args.pilot_cells or int(v2_contract["profiles"][args.profile]["cells"])
    base = (str(args.contract), str(args.elicitation_dir), calibrated["biases"], args.profile)
    learner_specs = [(r, f, g, cell) for r in design["regimes"] for f in design["families"][r]
                     for g in design["study_geometries"] for cell in range(cells_per_group)]
    learner_specs.sort(key=lambda s: "mlp" not in s[1])  # heaviest first, for load balance
    fixed_specs = [(r, g, cell) for r in design["regimes"] for g in design["study_geometries"]
                   for cell in range(cells_per_group)]
    jobs: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [(_learner_task, (*base, *s)) for s in learner_specs]
    jobs += [(_fixed_task, (*base, *s)) for s in fixed_specs]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(function, job) for function, job in jobs]
        for future in as_completed(futures):
            name, cell, timing = future.result()
            results.append(cell)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"done": len(results), "of": len(jobs), "task": name, **timing,
                                         "elapsed_seconds": round(time.perf_counter() - started, 1)}) + "\n")
    ordered = sorted(results, key=lambda c: (c["kind"], c["regime"], c.get("family", ""), c["geometry"], c["cell"]))
    cells_path = args.output_dir / "cells.jsonl"
    cells_path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in ordered), encoding="utf-8")
    summary: dict[str, Any] = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "cells_per_group": cells_per_group, "primary_measure": UTILITY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        "elicitation_sha256": _sha256(args.elicitation_dir / "elicitation.jsonl"),
        "world_calibration": calibrated["report"],
    }
    if all(p is not None for p in inputs.values()):
        records = {name: json.loads((ROOT / contract["inputs"][name]["record"]).read_text(encoding="utf-8"))
                   for name in ("utility", "prior_calibration")}
        summary.update(summarize(contract, v2_contract, records, o28.load_recorded(contract, v2_contract, inputs),
                                 ordered))
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": v2._host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    dispositions_ = summary.get("dispositions", {})
    print(json.dumps({"overall": dispositions_.get("overall"),
                      "attributions": {k: v["attribution"] for k, v in dispositions_.get("by_case", {}).items()},
                      "parity_failures": len(summary.get("parity_failures", [])),
                      "consistency_failures": len(summary.get("consistency_failures", [])), **receipt["outputs"]}))


if __name__ == "__main__":
    main()
