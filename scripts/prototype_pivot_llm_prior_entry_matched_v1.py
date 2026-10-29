#!/usr/bin/env python3
"""Run the precommitted Pivot LLM-prior entry-matched control v1.

The LLM prior's direction is crossed with its selectivity. A prior's direction
is its feature weights, and its selectivity is the fixed-rule entry rate it
produces in the world. Each matched condition keeps one prior's weights and
shifts only its intercept until its fixed-rule entry in the world equals the
other prior's. No LLM is loaded: the v2 calibrated prior is reused by hash.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_entry_matched_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_prior_pilot_v3 as pilot3  # noqa: E402


pilot1 = pilot3.pilot1
abm1 = pilot3.abm1
abm2 = pilot3.abm2
delayed = pilot3.delayed
persistence = pilot3.persistence
study2 = pilot3.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = pilot3.FIT_WORLDS
ABM_ARMS = pilot3.ABM_ARMS
FIELD_ARMS = pilot3.FIELD_ARMS
ALL_ARMS = tuple(arm for arm, _evidence in ABM_ARMS)
CONDITION_IDS = (
    "hand",
    "llm_selective",
    "hybrid_20",
    "hand_at_llm_entry",
    "hand_at_llm_entry_20",
    "llm_at_hand_entry",
)
DIRECTIONS = ("hand", "llm")
LEVELS = ("native", "llm_entry", "hand_entry")
# Each matched level names the direction whose native fixed-rule entry it targets.
MATCH_SOURCES = {"llm_entry": "llm", "hand_entry": "hand"}
OPENED_SEEDS = (
    pilot3.OPENED_SEEDS | set(range(20261023, 20261037)) | set(range(20261101, 20261106))
)
METRICS = (
    "cumulative_entry_rate",
    "exploratory_entry_rate",
    "productive_share_of_entrants",
    "probe_brier",
    "probe_auc",
    "active_failed_rate",
    "evidence_events_per_learner",
)


class CalibrationError(RuntimeError):
    """The entry match could not be bracketed or did not reach its tolerance."""


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("entry-matched contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_prior_entry_matched.v1":
        raise ValueError("unexpected entry-matched schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("entry-matched contract is not a precommit design")
    conditions = contract["conditions"]
    if [row["id"] for row in conditions] != list(CONDITION_IDS):
        raise ValueError("conditions changed")
    for row in conditions:
        if row["direction"] not in DIRECTIONS or row["level"] not in LEVELS:
            raise ValueError(f"unknown direction or level in {row['id']}")
        if row["level"] in MATCH_SOURCES and MATCH_SOURCES[row["level"]] == row["direction"]:
            raise ValueError("a matched level must target the other direction")
    if [tuple(row) for row in contract["abm_arms"]] != list(ABM_ARMS):
        raise ValueError("ABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    development = contract["profiles"]["development"]
    open_steps = int(development["steps"]) - int(development["burn_in_steps"])
    if int(contract["exploration_floor"]["open_steps"]) != open_steps:
        raise ValueError("exploration open steps must match the development profile")
    calibration = contract["entry_calibration"]
    seeds = [
        int(development["base_seed"]),
        int(contract["probe_cohort"]["base_seed"]),
        int(calibration["base_seed"]),
        int(calibration["probe_base_seed"]),
    ]
    if len(set(seeds)) != len(seeds) or set(seeds) & OPENED_SEEDS:
        raise ValueError("the entry-matched study needs fresh seed namespaces")
    if int(contract["profiles"]["smoke"]["base_seed"]) != int(development["base_seed"]):
        raise ValueError("smoke and development share the study seed namespace")
    if float(calibration["bracket"][0]) >= float(calibration["bracket"][1]):
        raise ValueError("calibration bracket must be increasing")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != pilot1.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this study")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    boundaries = contract["boundaries"]
    if boundaries.get("paid_api_calls") != 0 or boundaries.get("new_llm_calls") != 0:
        raise ValueError("the entry-matched study makes no LLM calls")


def load_llm_prior(contract: Mapping[str, Any]) -> dict[str, Any]:
    spec = contract["v2_calibrated_priors"]
    path = Path(spec["path"])
    if pilot1._sha256(path) != spec["sha256"]:
        raise RuntimeError("the v2 calibrated priors changed")
    prior = json.loads(path.read_text(encoding="utf-8"))["priors"][spec["prior_used"]]
    return {"weights": [float(v) for v in prior["weights"]], "bias": float(prior["bias"])}


def hand_prior() -> dict[str, Any]:
    weight, bias = pilot1.HAND_INITIAL_POLICY()
    return {"weights": [float(v) for v in weight.detach()], "bias": float(bias.detach())}


def direction_prior(direction: str, llm_prior: Mapping[str, Any]) -> dict[str, Any]:
    source = hand_prior() if direction == "hand" else llm_prior
    return {"weights": list(source["weights"]), "bias": float(source["bias"])}


def shifted(prior: Mapping[str, Any], shift: float) -> dict[str, Any]:
    return {"weights": list(prior["weights"]), "bias": float(prior["bias"]) + float(shift)}


def _condition(contract: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    return next(row for row in contract["conditions"] if row["id"] == condition_id)


def condition_prior(
    condition: Mapping[str, Any],
    llm_prior: Mapping[str, Any],
    shifts: Mapping[str, float],
) -> Mapping[str, Any] | None:
    """Return the prior for the slot; None keeps the hand rule's own policy."""

    if condition["level"] == "native":
        return None if condition["direction"] == "hand" else direction_prior("llm", llm_prior)
    return shifted(direction_prior(condition["direction"], llm_prior), shifts[condition["level"]])


# -- ABM tasks -----------------------------------------------------------------------


def _calibration_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The study contract with the calibration seed namespaces in place."""

    calibration = contract["entry_calibration"]
    reduced = copy.deepcopy(dict(contract))
    for profile in reduced["profiles"].values():
        profile["base_seed"] = int(calibration["base_seed"])
        profile["replicates"] = int(calibration["replicates"])
    reduced["probe_cohort"]["base_seed"] = int(calibration["probe_base_seed"])
    return reduced


def run_arms(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    world_id: str,
    prior: Mapping[str, Any] | None,
    epsilon: float,
    arms: Sequence[str],
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    runs = []
    for arm_id, evidence_regime in ABM_ARMS:
        if arm_id not in arms:
            continue
        adapter_class = pilot3.exploration_adapter(epsilon, config.base_seed, replicate)
        with pilot1.prior_slot(prior), pilot3.adapter_slot(adapter_class):
            run = abm2.run_arm(
                contract,
                config,
                replicate=replicate,
                visibility_id="immediate",
                world_id=world_id,
                arm_id=arm_id,
                evidence_regime=evidence_regime,
                probe=probe,
            )
        if len(adapter_class.created) != 1:
            raise RuntimeError("the exploration adapter was not used exactly once")
        runs.append(
            {
                "arm_id": arm_id,
                "evidence_regime": evidence_regime,
                "epsilon": epsilon,
                "exploratory_entry_rate": adapter_class.created[0].exploratory_entries
                / config.agent_count,
                **run,
            }
        )
    if study2._initial_policy is not pilot1.HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot was not restored")
    if persistence.PersistenceCommitAdapter is not pilot3.ORIGINAL_ADAPTER:
        raise RuntimeError("the adapter slot was not restored")
    return runs


def fixed_entry_task(args: tuple[Any, ...]) -> float:
    contract, profile, replicate, world_id, prior = args
    (run,) = run_arms(contract, profile, replicate, world_id, prior, 0.0, ("fixed_rule",))
    return float(run["cumulative_entry_rate"])


def hand_slot_nesting_task(args: tuple[Any, ...]) -> bool:
    """The hand prior passed through the slot with no shift equals the hand rule."""

    contract, profile, replicate, world_id = args
    native = run_arms(contract, profile, replicate, world_id, None, 0.0, ALL_ARMS[:1])
    through_slot = run_arms(contract, profile, replicate, world_id, hand_prior(), 0.0, ALL_ARMS[:1])
    return native == through_slot


def study_task(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    contract, profile, replicate, world_id, condition_id, llm_prior, shifts = args
    condition = _condition(contract, condition_id)
    config = abm1.study_config(contract, profile)
    share = float(condition["exploratory_share"])
    epsilon = pilot3.epsilon_for(share, config.steps - config.burn_in_steps)
    prior = condition_prior(condition, llm_prior, shifts[world_id])
    if prior is not None and prior["weights"] != direction_prior(condition["direction"], llm_prior)["weights"]:
        raise RuntimeError("a matched prior changed its direction's weights")
    runs = run_arms(contract, profile, replicate, world_id, prior, epsilon, ALL_ARMS)
    rows = [
        {
            "replicate": replicate,
            "world_id": world_id,
            "condition_id": condition_id,
            "direction": condition["direction"],
            "level": condition["level"],
            "exploratory_share": share,
            "prior_bias": None if prior is None else float(prior["bias"]),
            **run,
        }
        for run in runs
    ]
    if condition_id == "hand" and world_id == "linear_fit":
        fixed = next(row for row in runs if row["arm_id"] == "fixed_rule")
        pilot3.check_nesting(contract, config, replicate=replicate, fixed_run=fixed)
    return rows


# -- parallel map ----------------------------------------------------------------------


def _worker_init() -> None:
    torch.set_num_threads(1)
    pilot3._THREAD_LIMIT = threadpool_limits(limits=1)


class TaskMap:
    """Ordered map over tasks, serial or in a spawned process pool."""

    def __init__(self, workers: int) -> None:
        self.workers = int(workers)
        self.pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> TaskMap:
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    def __call__(self, function: Any, tasks: Sequence[Any]) -> list[Any]:
        if self.pool is None:
            torch.set_num_threads(1)
            with threadpool_limits(limits=1):
                return [function(task) for task in tasks]
        return list(self.pool.map(function, tasks))


# -- entry calibration -----------------------------------------------------------------


def calibrate_entry(
    contract: Mapping[str, Any],
    profile: str,
    llm_prior: Mapping[str, Any],
    task_map: TaskMap,
) -> dict[str, Any]:
    """Bisect each matched intercept shift on the calibration replicates, per world."""

    spec = contract["entry_calibration"]
    cal = _calibration_contract(contract)
    replicates = range(int(spec["replicates"]))
    targets = [(world, level) for world in FIT_WORLDS for level in MATCH_SOURCES]

    def mean_entries(points: Sequence[tuple[str, Mapping[str, Any] | None]]) -> list[float]:
        tasks = [(cal, profile, r, world, prior) for world, prior in points for r in replicates]
        values = task_map(fixed_entry_task, tasks)
        width = len(replicates)
        return [float(np.mean(values[i * width : (i + 1) * width])) for i in range(len(points))]

    nesting = task_map(
        hand_slot_nesting_task, [(cal, profile, r, world) for world in FIT_WORLDS for r in replicates]
    )
    if not all(nesting):
        raise RuntimeError("the hand prior through the slot does not equal the hand rule")

    def source_prior(direction: str) -> Mapping[str, Any] | None:
        return None if direction == "hand" else direction_prior("llm", llm_prior)

    source_entries = mean_entries(
        [(world, source_prior(MATCH_SOURCES[level])) for world, level in targets]
    )
    target_entry = dict(zip(targets, source_entries, strict=True))

    def matched_prior(level: str, shift: float) -> Mapping[str, Any]:
        direction = "hand" if MATCH_SOURCES[level] == "llm" else "llm"
        return shifted(direction_prior(direction, llm_prior), shift)

    low, high = (float(value) for value in spec["bracket"])
    lows = {key: low for key in targets}
    highs = {key: high for key in targets}
    edges = mean_entries(
        [(world, matched_prior(level, low)) for world, level in targets]
        + [(world, matched_prior(level, high)) for world, level in targets]
    )
    for index, key in enumerate(targets):
        at_low, at_high = edges[index], edges[len(targets) + index]
        if not at_low <= target_entry[key] <= at_high:
            raise CalibrationError(
                f"entry target {target_entry[key]} not bracketed for {key}: [{at_low}, {at_high}]"
            )
    best = {key: (math.inf, 0.0, math.nan) for key in targets}
    for _iteration in range(int(spec["bisection_iterations"])):
        mids = {key: 0.5 * (lows[key] + highs[key]) for key in targets}
        entries = mean_entries([(world, matched_prior(level, mids[(world, level)])) for world, level in targets])
        for key, entry in zip(targets, entries, strict=True):
            gap = abs(entry - target_entry[key])
            if gap < best[key][0]:
                best[key] = (gap, mids[key], entry)
            if entry < target_entry[key]:
                lows[key] = mids[key]
            else:
                highs[key] = mids[key]
    tolerance = float(spec["in_sample_relative_tolerance"])
    shifts: dict[str, dict[str, float]] = {world: {} for world in FIT_WORLDS}
    records = []
    for world, level in targets:
        _gap, shift, achieved = best[(world, level)]
        target = target_entry[(world, level)]
        ratio = achieved / target
        if abs(ratio - 1.0) > tolerance:
            raise CalibrationError(f"in-sample entry ratio {ratio} misses the tolerance for {world}/{level}")
        shifts[world][level] = shift
        records.append(
            {
                "world_id": world,
                "level": level,
                "matched_direction": "hand" if MATCH_SOURCES[level] == "llm" else "llm",
                "source_direction": MATCH_SOURCES[level],
                "intercept_shift": shift,
                "target_entry": target,
                "achieved_entry": achieved,
                "in_sample_ratio": ratio,
            }
        )
    return {
        "status": "ENTRY_CALIBRATION_COMPLETE",
        "replicates": len(replicates),
        "hand_slot_nesting": "PASS_EXACT",
        "records": records,
        "shifts": shifts,
    }


def run_study(
    contract: Mapping[str, Any],
    profile: str,
    llm_prior: Mapping[str, Any],
    workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = abm1.study_config(contract, profile)
    with TaskMap(workers) as task_map:
        calibration = calibrate_entry(contract, profile, llm_prior, task_map)
        tasks = [
            (contract, profile, replicate, world_id, condition_id, llm_prior, calibration["shifts"])
            for replicate in range(config.replicates)
            for world_id in FIT_WORLDS
            for condition_id in CONDITION_IDS
        ]
        results = task_map(study_task, tasks)
    return calibration, [row for rows in results for row in rows]


# -- decisions -------------------------------------------------------------------------


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {
        (int(row["replicate"]), row["world_id"], row["condition_id"], row["arm_id"]): row
        for row in rows
    }


def _values(indexed, replicates, world, condition, arm, metric) -> np.ndarray:  # noqa: ANN001
    return np.asarray(
        [float(indexed[(r, world, condition, arm)][metric]) for r in replicates], dtype=np.float64
    )


def interval(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    mean = float(np.mean(values))
    half = float(student_t.ppf(0.975, count - 1) * np.std(values, ddof=1) / math.sqrt(count))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": count}


def precision_outcome(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    contract: Mapping[str, Any],
    *,
    first: str,
    second: str,
    world: str,
    arm: str,
) -> dict[str, Any]:
    rule = contract["decision_contract"]["precision"]
    minimum = float(rule["improvement_at_least"])
    required = int(rule["improved_replicates_at_least"])
    metric = "productive_share_of_entrants"
    gap = _values(indexed, replicates, world, first, arm, metric) - _values(
        indexed, replicates, world, second, arm, metric
    )
    first_better = abm1._judge(gap, minimum, required)
    second_better = abm1._judge(-gap, minimum, required)
    if first_better["pass"]:
        outcome = "FIRST_BETTER"
    elif second_better["pass"]:
        outcome = "SECOND_BETTER"
    else:
        outcome = "NO_CLEAR_DIFFERENCE"
    return {
        "outcome": outcome,
        "first_minus_second": first_better["improvement"],
        "first_positive_replicates": first_better["positive_replicates"],
        "second_positive_replicates": second_better["positive_replicates"],
        "interval_95": interval(gap),
    }


def learning_noninferior(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    contract: Mapping[str, Any],
    *,
    condition: str,
    reference: str,
    world: str,
    arm: str,
) -> dict[str, Any]:
    margin = float(contract["decision_contract"]["learning_noninferiority"]["margin"])
    gap = _values(indexed, replicates, world, reference, arm, "probe_brier") - _values(
        indexed, replicates, world, condition, arm, "probe_brier"
    )
    bounds = interval(gap)
    return {
        "reference_minus_condition_probe_brier": bounds,
        "pass": bounds["low"] >= margin,
        "mean_only_rule_pass": bounds["mean"] >= margin,
    }


def _disposition(precision: str, noninferior: bool) -> str:
    if precision == "FIRST_BETTER" and noninferior:
        return "LLM_DOMINATES"
    if precision == "FIRST_BETTER":
        return "SELECTION_ONLY"
    if noninferior:
        return "LEARNING_ONLY"
    return "NEITHER"


def _count_label(count: int, total: int) -> str:
    if count == total:
        return "ALL_CELLS"
    if count * 2 > total:
        return "MAJORITY"
    if count > 0:
        return "MINORITY"
    return "NO_CELL"


def _ranking_comparison(indexed, replicates, contract, *, first, second, arms) -> dict[str, Any]:  # noqa: ANN001
    cells = {
        f"{world}/{arm}": precision_outcome(
            indexed, replicates, contract, first=first, second=second, world=world, arm=arm
        )
        for world in FIT_WORLDS
        for arm in arms
    }
    counts = {
        outcome: sum(cell["outcome"] == outcome for cell in cells.values())
        for outcome in ("FIRST_BETTER", "SECOND_BETTER", "NO_CLEAR_DIFFERENCE")
    }
    return {"first": first, "second": second, "counts": counts, "cells": cells}


def _dominance_comparison(indexed, replicates, contract, *, condition, reference) -> dict[str, Any]:  # noqa: ANN001
    cells = {}
    for world in FIT_WORLDS:
        for arm in FIELD_ARMS:
            precision = precision_outcome(
                indexed, replicates, contract, first=condition, second=reference, world=world, arm=arm
            )
            learning = learning_noninferior(
                indexed, replicates, contract, condition=condition, reference=reference, world=world, arm=arm
            )
            cells[f"{world}/{arm}"] = {
                "disposition": _disposition(precision["outcome"], learning["pass"]),
                "mean_only_disposition": _disposition(precision["outcome"], learning["mean_only_rule_pass"]),
                "precision": precision,
                "learning": learning,
            }
    dominating = sorted(key for key, cell in cells.items() if cell["disposition"] == "LLM_DOMINATES")
    mean_only = sorted(key for key, cell in cells.items() if cell["mean_only_disposition"] == "LLM_DOMINATES")
    selecting = sorted(key for key, cell in cells.items() if cell["precision"]["outcome"] == "FIRST_BETTER")
    return {
        "condition": condition,
        "reference": reference,
        "dominating_cells": dominating,
        "dominance": _count_label(len(dominating), len(cells)),
        "mean_only_dominating_cells": mean_only,
        "precision_better_cells": selecting,
        "precision_worse_cells": sorted(
            key for key, cell in cells.items() if cell["precision"]["outcome"] == "SECOND_BETTER"
        ),
        "cells": cells,
    }


def entry_match_check(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    low, high = (float(value) for value in contract["entry_calibration"]["out_of_sample_ratio_bounds"])
    pairs = (
        ("hand_at_llm_entry", "llm_selective"),
        ("hand_at_llm_entry_20", "hybrid_20"),
        ("llm_at_hand_entry", "hand"),
    )
    registered = {"hand_at_llm_entry", "llm_at_hand_entry"}
    records, passed = [], True
    for matched, source in pairs:
        for world in FIT_WORLDS:
            ratio = float(
                np.mean(_values(indexed, replicates, world, matched, "fixed_rule", "cumulative_entry_rate"))
                / np.mean(_values(indexed, replicates, world, source, "fixed_rule", "cumulative_entry_rate"))
            )
            checked = matched in registered
            ok = low <= ratio <= high
            passed = passed and (ok or not checked)
            records.append(
                {
                    "matched": matched,
                    "source": source,
                    "world_id": world,
                    "fixed_rule_entry_ratio": ratio,
                    "registered_check": checked,
                    "within_bounds": ok,
                }
            )
    return {"status": "PASS" if passed else "FAIL", "bounds": [low, high], "records": records}


def _c22_status(comparison: Mapping[str, Any]) -> str:
    count = len(comparison["precision_better_cells"])
    if count == 0:
        return "C22_PRECISION_IS_SELECTIVITY"
    if count * 2 > len(comparison["cells"]):
        return "C22_PRECISION_SURVIVES_ENTRY_MATCHING"
    return "C22_PRECISION_SURVIVES_IN_MINORITY"


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = _index(rows)
    replicates = sorted({int(row["replicate"]) for row in rows})
    check = entry_match_check(indexed, replicates, contract)
    if check["status"] != "PASS":
        return {"status": "ENTRY_MATCH_FAILED_STOP", "entry_match_check": check}
    matched_c22 = _dominance_comparison(
        indexed, replicates, contract, condition="hybrid_20", reference="hand_at_llm_entry_20"
    )
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "entry_match_check": check,
        "ranking_at_llm_selectivity_fixed_rule": _ranking_comparison(
            indexed, replicates, contract, first="llm_selective", second="hand_at_llm_entry", arms=("fixed_rule",)
        ),
        "ranking_at_llm_selectivity_learners": _ranking_comparison(
            indexed, replicates, contract, first="llm_selective", second="hand_at_llm_entry", arms=FIELD_ARMS
        ),
        "ranking_at_hand_selectivity": _ranking_comparison(
            indexed, replicates, contract, first="llm_at_hand_entry", second="hand", arms=ALL_ARMS
        ),
        "c22_against_entry_matched_hand": {**matched_c22, "status": _c22_status(matched_c22)},
        "c22_replication_against_hand": _dominance_comparison(
            indexed, replicates, contract, condition="hybrid_20", reference="hand"
        ),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["condition_id"], row["arm_id"])].append(row)
    cells = []
    for world in FIT_WORLDS:
        for condition in CONDITION_IDS:
            for arm in ALL_ARMS:
                cell_rows = grouped[(world, condition, arm)]
                means = {}
                for metric in METRICS:
                    values = [row[metric] for row in cell_rows if row[metric] is not None]
                    means[metric] = round(float(np.mean(values)), 6) if values else None
                cells.append(
                    {
                        "world_id": world,
                        "condition_id": condition,
                        "arm_id": arm,
                        "replicates": len(cell_rows),
                        "means": means,
                    }
                )
    config = abm1.study_config(contract, profile)
    required = int(contract["decision_contract"]["precision"]["improved_replicates_at_least"])
    decisions = (
        {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        if config.replicates < required
        else decide(rows, contract)
    )
    return {
        "schema_version": "pivot_llm_prior_entry_matched_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "calibration": {key: value for key, value in calibration.items() if key != "shifts"},
        "cells": cells,
        "abm_runs": len(rows),
        "decisions": decisions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot LLM-prior entry-matched control scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "## Entry calibration",
        "",
        "| World | Level | Matched direction | Intercept shift | Target entry | Achieved |",
        "|---|---|---|---:|---:|---:|",
    ]
    for record in summary["calibration"]["records"]:
        lines.append(
            f"| {record['world_id']} | {record['level']} | {record['matched_direction']} | "
            f"{record['intercept_shift']:+.4f} | {record['target_entry']:.4f} | {record['achieved_entry']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cells",
            "",
            "| World | Condition | Arm | Entry | Exploratory | Entrant precision | Probe Brier |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in summary["cells"]:
        m = cell["means"]
        lines.append(
            f"| {cell['world_id']} | {cell['condition_id']} | {cell['arm_id']} | "
            f"{m['cumulative_entry_rate']:.4f} | {m['exploratory_entry_rate']:.4f} | "
            f"{m['productive_share_of_entrants']:.4f} | {m['probe_brier']:.4f} |"
        )
    decisions = summary["decisions"]
    lines.extend(["", f"Decision status: `{decisions['status']}`", ""])
    if decisions["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE":
        for key in (
            "ranking_at_llm_selectivity_fixed_rule",
            "ranking_at_llm_selectivity_learners",
            "ranking_at_hand_selectivity",
        ):
            entry = decisions[key]
            lines.append(f"- {key} ({entry['first']} vs {entry['second']}): {entry['counts']}")
        matched = decisions["c22_against_entry_matched_hand"]
        replication = decisions["c22_replication_against_hand"]
        lines.append(
            f"- C22 against the entry-matched hand rule: `{matched['status']}`, precision better in "
            f"{matched['precision_better_cells'] or '-'}, dominating {matched['dominating_cells'] or '-'}"
        )
        lines.append(
            f"- C22 replication against the hand rule: dominating {replication['dominating_cells'] or '-'}"
            f" (mean-only rule: {replication['mean_only_dominating_cells'] or '-'})"
        )
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "calibration.json": json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in files.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_llm_prior_entry_matched_receipt.v1",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_sha256": pilot1._sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {name: pilot1._sha256(output_dir / name) for name in files},
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(
        output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    config = abm1.study_config(contract, "development")
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": pilot1._sha256(config_path),
        "conditions": list(CONDITION_IDS),
        "epsilon_by_condition": {
            row["id"]: pilot3.epsilon_for(row["exploratory_share"], config.steps - config.burn_in_steps)
            for row in contract["conditions"]
        },
        "source_bindings": delayed.verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error("run requires a full 40-character lowercase --precommit-sha")
    if args.output_dir is None:
        parser.error("run requires --output-dir")
    precommit = delayed.verify_precommit(args.precommit_sha, contract)
    llm_prior = load_llm_prior(contract)
    calibration, rows = run_study(contract, args.profile, llm_prior, workers=args.workers)
    summary = {
        **summarize(rows, calibration, contract, args.profile),
        "precommit_sha": args.precommit_sha,
    }
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        calibration=calibration,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
