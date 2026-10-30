#!/usr/bin/env python3
"""Run the precommitted Pivot learning-strength study v1 (O39).

Under the peer-outcome estimator of recommended experiment 4, Study 2's
imitative arm answers a neighbor's outcome far more strongly than the
OpenAlex coauthor data allow (C93). This script reruns Study 2 unchanged over
a grid of learning rates, at the tracked outcome delay and at a three-step
delay, and measures every imitative arm with that frozen estimator. It finds
the largest rate on the grid whose visible outcome gap the data allow, and
reads there Study 2's registered comparison (R5): the failed pivots of the
imitative arm against the frozen arm.

No empirical row is read. The data bound is a number in the recorded result
of experiment 4.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_learning_strength_v1.yaml"
SCHEMA = "pivot_learning_strength.v1"
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
STUDY2_ROLE = "STUDY2_SCRIPT_RUN_UNCHANGED_THROUGH_THE_E4_SLOTS"
TRACKED_ROLE = "STUDY2_TRACKED_REPLICATE_RUNS_AND_COMPARISONS_FOR_PARITY"
E4_SCRIPT_ROLE = "E4_SCRIPT_SLOTS_ROWS_AND_ESTIMATOR_REUSED_BY_HASH_BOUND_IMPORT"
E4_CONTRACT_ROLE = "E4_CONTRACT_ESTIMATOR_SETTINGS_SCENARIOS_AND_STUDY2_BINDINGS"
E4_RECORD_ROLE = "E4_RECORD_DATA_BOUND_AND_SIMULATION_ROWS_FOR_PARITY"
IDENTITY = ["o39_summary.json", "o39_receipt.json"]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
ARMS = ("frozen", "imitative", "cautionary")
ESTIMATED_ARMS = ("frozen", "imitative")
OUTCOMES = ("failed_pivot_rate", "productive_pivot_rate")
RATE_KEYS = ("pivot_rate", "failed_pivot_rate", "productive_pivot_rate", "productive_share_of_pivots")
LOOP_SCENARIOS = ("baseline", "interdisciplinary_seed_grants")
E4_PARITY_TOLERANCE = 1e-5


class StrengthError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StrengthError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(contract: Mapping[str, Any], role: str) -> Path:
    matches = [b for b in contract["source_bindings"] if b["role"] == role]
    require(len(matches) == 1, f"exactly one source binding must have role {role}")
    return ROOT / matches[0]["path"]


def _pointer(document: Any, pointer: str) -> Any:
    value = document
    for part in pointer.strip("/").split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def registered_rates() -> list[float]:
    """The tracked rate halved seven times; every value is exact in binary."""

    return [1.5 / 2**k for k in range(8)]


# -- contract ----------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require(isinstance(contract, dict), "the contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == SCHEMA, "unexpected contract schema")
    require(contract.get("status") == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION", "the contract is not a precommit design")
    for binding in contract["source_bindings"]:
        require(_sha256(ROOT / binding["path"]) == binding["sha256"], f"source binding drifted: {binding['path']}")
    simulation = contract["simulation"]
    require(simulation["learning_rates"] == registered_rates(), "the learning rates are 1.5 halved seven times")
    require(simulation["arms"] == list(ARMS), "the arms are frozen, imitative, and cautionary")
    require(simulation["estimated_arms"] == list(ESTIMATED_ARMS), "the estimator measures the frozen and imitative arms")
    require(simulation["outcome_delays"] == [0, 3], "the outcome delays are 0 and 3")
    require(int(simulation["replicates"]) == 30, "Study 2 has 30 replicates")
    e4_contract = yaml.safe_load(_binding(contract, E4_CONTRACT_ROLE).read_text(encoding="utf-8"))
    require(simulation["scenarios"] == e4_contract["simulation"]["scenarios"], "the scenarios are E4's, in E4's order")
    require(simulation["outcome_delays"] == e4_contract["simulation"]["outcome_delays"], "the delays are E4's")
    calibration = contract["calibration"]
    record = json.loads(_binding(contract, E4_RECORD_ROLE).read_text(encoding="utf-8"))
    for key in ("primary", "secondary"):
        bound = calibration[key]
        require(_pointer(record, bound["pointer"]) == bound["value"], f"the {key} data bound differs from the E4 record")
    require(calibration["quantity"] == "state/outcome_gap_visible", "the calibration quantity is the visible outcome gap")
    require(calibration["primary"]["value"] > calibration["secondary"]["value"], "the primary bound is the upper bound")
    decision = contract["decision"]
    require(decision["primary_delay"] == 3, "the headline delay is 3")
    require(decision["material_failed_pivot_delta"] == 0.05, "the material failed-pivot delta is 0.05")
    require(decision["loop_scenarios"] == list(LOOP_SCENARIOS), "R5's loop is read in baseline and grants")
    require(contract["execution"]["identity_artifacts"] == IDENTITY, "identity artifacts differ")
    require(contract["execution"]["thread_variables"] == list(THREAD_VARIABLES), "the thread variables differ")
    boundary = contract["boundary"]
    require(boundary["empirical_rows_read"] == 0 and boundary["study2_code_changed"] is False, "the boundary differs")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    require(FULL_SHA_PATTERN.fullmatch(commit_sha) is not None, "run requires a full 40-character lowercase Git SHA")
    head = _git("rev-parse", "HEAD")
    require(head == commit_sha, f"precommit SHA is not HEAD: {commit_sha} != {head}")
    paths = [str(path) for path in contract["precommit"]["precommit_paths"]]
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *paths)
    require(not dirty, f"precommit paths are dirty:\n{dirty}")
    tracked = set(_git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    missing = sorted(set(paths) - tracked)
    require(not missing, f"precommit paths missing from commit: {missing}")
    return {
        "commit_sha": commit_sha,
        "precommit_paths": paths,
        "source_bindings": [{"path": b["path"], "sha256": b["sha256"]} for b in contract["source_bindings"]],
    }


# -- modules by hash-bound path ----------------------------------------------------------


def load_e4(contract: Mapping[str, Any]) -> ModuleType:
    name = "pivot_peer_outcome_contrast_v1_for_o39"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _binding(contract, E4_SCRIPT_ROLE))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules(contract: Mapping[str, Any]) -> tuple[ModuleType, dict[str, Any], ModuleType]:
    """E4's script, its validated contract, and Study 2 loaded through it."""

    e4 = load_e4(contract)
    e4_contract = e4.load_contract(_binding(contract, E4_CONTRACT_ROLE))
    study2 = e4.load_study2(e4_contract)
    require(Path(study2.__file__).resolve() == _binding(contract, STUDY2_ROLE).resolve(), "E4 loaded another Study 2")
    return e4, e4_contract, study2


# -- Study 2 over the grid ---------------------------------------------------------------


def simulation_tasks(contract: Mapping[str, Any]) -> list[tuple[str, int | None, str, int]]:
    """(arm, rate index, scenario, delay); the frozen arm does not learn, so it has no rate."""

    spec = contract["simulation"]
    tasks: list[tuple[str, int | None, str, int]] = []
    for scenario in spec["scenarios"]:
        for delay in spec["outcome_delays"]:
            tasks.append(("frozen", None, scenario, delay))
            for arm in ARMS[1:]:
                tasks.extend((arm, k, scenario, delay) for k in range(len(spec["learning_rates"])))
    return tasks


def run_task(e4: ModuleType, study2: ModuleType, scenarios: Mapping[str, Any], task: Sequence[Any], *, rates: Sequence[float],
             replicates: int) -> tuple[list[dict], list[dict]]:
    arm, k, scenario, delay = task
    config = study2.LearningStudyConfig() if k is None else study2.LearningStudyConfig(learning_rate=rates[k])
    runs, logs = [], []
    for replicate in range(replicates):
        run, log = e4.run_logged_arm(study2, scenario=scenarios[scenario], config=config, replicate=replicate, mode=arm, delay=delay)
        runs.append({key: run[key] for key in e4.PARITY_KEYS})
        logs.append(log)
    return runs, logs


_SIMULATION: dict[str, Any] = {}


def _init_simulation(contract_path: str) -> None:
    import torch

    torch.set_num_threads(1)
    contract = load_contract(Path(contract_path))
    e4, _e4_contract, study2 = load_modules(contract)
    _SIMULATION.update(e4=e4, study2=study2, scenarios={s.name: s for s in study2.study1.study_scenarios()},
                       rates=list(contract["simulation"]["learning_rates"]), replicates=int(contract["simulation"]["replicates"]))


def _simulate(task: tuple[str, int | None, str, int]) -> tuple[tuple, list[dict], list[dict]]:
    s = _SIMULATION
    runs, logs = run_task(s["e4"], s["study2"], s["scenarios"], task, rates=s["rates"], replicates=s["replicates"])
    return task, runs, logs


def run_simulation(contract: Mapping[str, Any], contract_path: Path, *, workers: int,
                   tasks: Sequence[tuple[str, int | None, str, int]] | None = None) -> dict[tuple, dict[str, list]]:
    tasks = list(simulation_tasks(contract) if tasks is None else tasks)
    if workers <= 1:
        _init_simulation(str(contract_path))
        outputs = [_simulate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_simulation,
                                 initargs=(str(contract_path),)) as pool:
            outputs = list(pool.map(_simulate, tasks))
    return {tuple(task): {"runs": runs, "logs": logs} for task, runs, logs in outputs}


def simulation_checks(contract: Mapping[str, Any], study2: ModuleType, tracked: Mapping[str, Any], results: Mapping[tuple, Any],
                      e4: ModuleType) -> dict[str, Any]:
    """Parity before any estimate: the tracked runs and comparisons at the tracked rate, and the frozen arm across delays."""

    spec = contract["simulation"]
    config = study2.LearningStudyConfig()
    tracked_config = {k: v for k, v in tracked["config"].items()}
    require(tracked_config == {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()},
            "the Study 2 config differs from the tracked one")
    require(spec["learning_rates"][0] == config.learning_rate, "the grid starts at the tracked rate")
    checks = {"tracked_runs_checked": 0, "tracked_runs_equal": 0, "frozen_delay_3_runs_equal": 0, "tracked_comparisons_equal": 0}
    for scenario in spec["scenarios"]:
        for arm in ARMS:
            k = None if arm == "frozen" else 0
            runs = results[(arm, k, scenario, 0)]["runs"]
            recorded = tracked["scenarios"][scenario][arm]["replicate_runs"]
            for run, row in zip(runs, recorded, strict=True):
                checks["tracked_runs_checked"] += 1
                checks["tracked_runs_equal"] += int(all(run[key] == row[key] for key in e4.PARITY_KEYS))
        frozen = results[("frozen", None, scenario, 0)]["logs"]
        delayed = results[("frozen", None, scenario, 3)]["logs"]
        checks["frozen_delay_3_runs_equal"] += sum(int(a == b) for a, b in zip(frozen, delayed, strict=True))
    computed = comparisons(contract, study2, results)
    for row in tracked["comparisons"]:
        arm = row["arm"].split(" - ")[0]
        mine = computed[f"{arm}/0/{row['scenario']}/delay_0/{row['outcome_field']}"]
        keys = ("paired_deltas", "mean", "std", "empirical_interval_95", "mean_ci95")
        checks["tracked_comparisons_equal"] += int(all(mine[key] == row[key] for key in keys))
    replicates = int(spec["replicates"])
    checks["tracked_comparisons_checked"] = len(tracked["comparisons"])
    checks["passes"] = (
        checks["tracked_runs_checked"] == checks["tracked_runs_equal"] == len(spec["scenarios"]) * len(ARMS) * replicates
        and checks["frozen_delay_3_runs_equal"] == len(spec["scenarios"]) * replicates
        and checks["tracked_comparisons_equal"] == checks["tracked_comparisons_checked"] > 0
    )
    return checks


def comparisons(contract: Mapping[str, Any], study2: ModuleType, results: Mapping[tuple, Any]) -> dict[str, dict[str, Any]]:
    """Study 2's paired comparison of each learning arm with the frozen arm, by its own distribution function."""

    spec = contract["simulation"]
    out: dict[str, dict[str, Any]] = {}
    for scenario in spec["scenarios"]:
        for delay in spec["outcome_delays"]:
            frozen = results[("frozen", None, scenario, delay)]["runs"]
            for arm in ARMS[1:]:
                for k, rate in enumerate(spec["learning_rates"]):
                    learning = results[(arm, k, scenario, delay)]["runs"]
                    for outcome in OUTCOMES:
                        deltas = [own[outcome] - base[outcome] for base, own in zip(frozen, learning, strict=True)]
                        out[f"{arm}/{k}/{scenario}/delay_{delay}/{outcome}"] = {
                            "arm": arm, "rate": rate, "scenario": scenario, "delay": delay, "outcome_field": outcome,
                            "paired_deltas": [round(float(delta), 8) for delta in deltas],
                            **study2._distribution(deltas),
                        }
    return out


def arm_rates(contract: Mapping[str, Any], results: Mapping[tuple, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for (arm, k, scenario, delay), result in sorted(results.items(), key=lambda item: str(item[0])):
        rate = None if k is None else contract["simulation"]["learning_rates"][k]
        label = f"{arm}/{'none' if k is None else k}/{scenario}/delay_{delay}"
        out[label] = {"rate": rate, **{key: round(float(np.mean([run[key] for run in result["runs"]])), 6) for key in RATE_KEYS}}
    return out


# -- the E4 estimator over the grid ------------------------------------------------------


def problem_key(arm: str, k: int | None, scenario: str, delay: int) -> str:
    return f"{arm}/{'none' if k is None else k}/{scenario}/delay_{delay}"


def estimation_problems(contract: Mapping[str, Any], e4: ModuleType, e4_contract: Mapping[str, Any], study2: ModuleType,
                        results: Mapping[tuple, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """E4's three models per estimated arm, with E4's replicate draws for every rate of a scenario and delay."""

    estimator = e4_contract["estimator"]
    settings = e4._settings(estimator)
    spec = contract["simulation"]
    replicates = int(spec["replicates"])
    seed, draws = int(estimator["bootstrap"]["seed"]), int(estimator["bootstrap"]["draws"])
    config = study2.LearningStudyConfig()
    models_spec = (("exposure", ["beta:exposed"]), ("state", list(e4.CONTRASTS)), ("entry_order", ["beta:entry_time"]))
    problems: dict[str, Any] = {}
    points: dict[str, Any] = {}
    full_fits = {"fits": 0, "unconverged": []}
    for s_index, scenario in enumerate(spec["scenarios"]):
        for delay in spec["outcome_delays"]:
            draws_matrix = e4.multiplicities(replicates, draws, seed, 1, s_index, delay)
            entries = [("frozen", None)] + [("imitative", k) for k in range(len(spec["learning_rates"]))]
            for arm, k in entries:
                rows = e4.simulation_rows(results[(arm, k, scenario, delay)]["logs"], delay=delay, burn_in=int(config.burn_in_steps),
                                          steps=int(config.steps))
                cells = e4.simulation_cells(rows, estimator)
                models = []
                for model_key, quantities in models_spec:
                    model_cells = cells[model_key]
                    fit = e4.fit_logit(model_cells, **settings)
                    full_fits["fits"] += 1
                    if not fit["converged"]:
                        full_fits["unconverged"].append(f"{problem_key(arm, k, scenario, delay)}/{model_key}")
                    models.append({"arm": arm, "model": model_key, "cells": model_cells, "beta": fit["beta"],
                                   "reference": model_cells.state_events.get("reference", "P_F"), "quantities": quantities,
                                   "converged": bool(fit["converged"])})
                key = problem_key(arm, k, scenario, delay)
                problems[key] = {"models": models, "multiplicities": draws_matrix}
                points[key] = [e4._point_values(m) for m in models]
    return problems, points, full_fits


def _transport(problems: Mapping[str, Any]) -> dict[str, Any]:
    """Problems with each Cells as a plain mapping, so spawned workers rebuild them with their own E4 module."""

    return {
        key: {"multiplicities": problem["multiplicities"],
              "models": [{**{k: v for k, v in model.items() if k != "cells"}, "cells": dict(vars(model["cells"]))} for model in problem["models"]]}
        for key, problem in problems.items()
    }


_BOOTSTRAP: dict[str, Any] = {}


def _init_bootstrap(contract_path: str, problems: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    contract = load_contract(Path(contract_path))
    e4 = load_e4(contract)
    rebuilt = {
        key: {"multiplicities": problem["multiplicities"],
              "models": [{**model, "cells": e4.Cells(**model["cells"])} for model in problem["models"]]}
        for key, problem in problems.items()
    }
    e4._init_worker(rebuilt, settings)
    _BOOTSTRAP["e4"] = e4


def _bootstrap_chunk(task: tuple[str, int, int]) -> tuple[str, int, list[list[float]], int]:
    return _BOOTSTRAP["e4"]._run_chunk(task)


def run_bootstrap(contract_path: Path, e4: ModuleType, problems: Mapping[str, Any], settings: Mapping[str, Any], *, workers: int,
                  chunk: int = 50) -> dict[str, dict[str, Any]]:
    """E4's bootstrap; with workers, each spawned worker loads E4 by its bound path."""

    if workers <= 1:
        return e4.bootstrap(problems, settings, workers=1, chunk=chunk)
    tasks = []
    for key, problem in problems.items():
        count = problem["multiplicities"].shape[0]
        tasks.extend((key, start, min(start + chunk, count)) for start in range(0, count, chunk))
    results: dict[str, list] = {key: [] for key in problems}
    failures = dict.fromkeys(problems, 0)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_bootstrap,
                             initargs=(str(contract_path), _transport(problems), dict(settings))) as pool:
        for key, start, values, failed in pool.map(_bootstrap_chunk, tasks):
            results[key].append((start, values))
            failures[key] += failed
    return {
        key: {"draws": np.asarray([v for _start, chunk_values in sorted(results[key], key=lambda item: item[0]) for v in chunk_values],
                                  dtype=np.float64),
              "unconverged_fits": failures[key]}
        for key in problems
    }


def estimation_report(e4: ModuleType, problems: Mapping[str, Any], points: Mapping[str, Any], draws: Mapping[str, Any]) -> dict[str, Any]:
    """E4's report per arm and rate; learning-attributable rows subtract the frozen arm on the same draws, as in E4."""

    report: dict[str, Any] = {}
    columns: dict[str, dict[str, int]] = {}
    for key, problem in problems.items():
        values = draws[key]["draws"]
        rows: dict[str, Any] = {}
        columns[key] = {}
        column = 0
        for model, point in zip(problem["models"], points[key], strict=True):
            for quantity, value in zip(model["quantities"], point, strict=True):
                name = quantity[5:] if quantity.startswith("beta:") else quantity
                label = f"{model['model']}/{name}"
                defined = model["converged"] and e4._quantity_defined(model, quantity)
                rows[label] = e4.summarize_draws(value if defined else None, values[:, column])
                columns[key][label] = column
                column += 1
        state = next(model for model in problem["models"] if model["model"] == "state")
        report[key] = {"rows": rows, "state_events": dict(state["cells"].state_events), "unconverged_fits": draws[key]["unconverged_fits"]}
    for key in problems:
        arm, k, scenario, delay = key.split("/")
        if arm == "frozen":
            continue
        base_key = problem_key("frozen", None, scenario, int(delay.removeprefix("delay_")))
        attributable: dict[str, Any] = {}
        for label, own in report[key]["rows"].items():
            base = report[base_key]["rows"][label]
            if own["mean"] is None or base["mean"] is None:
                attributable[label] = e4.summarize_draws(None, np.array([]))
                continue
            diff = draws[key]["draws"][:, columns[key][label]] - draws[base_key]["draws"][:, columns[base_key][label]]
            attributable[label] = e4.summarize_draws(own["mean"] - base["mean"], diff)
        report[key]["learning_attributable"] = attributable
    return report


def e4_parity(contract: Mapping[str, Any], record: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """At the tracked rate the estimator rows must equal E4's recorded simulation rows."""

    compared, largest, mismatched = 0, 0.0, []
    for scenario in contract["simulation"]["scenarios"]:
        for delay in contract["simulation"]["outcome_delays"]:
            recorded = record["simulation"][f"simulation/{scenario}/delay_{delay}"]
            pairs = [(recorded["by_arm"]["frozen"], report[problem_key("frozen", None, scenario, delay)]["rows"]),
                     (recorded["by_arm"]["imitative"], report[problem_key("imitative", 0, scenario, delay)]["rows"]),
                     (recorded["learning_attributable"]["imitative"], report[problem_key("imitative", 0, scenario, delay)]["learning_attributable"])]
            for theirs, mine in pairs:
                for label, row in theirs.items():
                    for field in ("mean", "low", "high"):
                        compared += 1
                        a, b = row[field], mine[label][field]
                        if a is None or b is None:
                            if a is not b:
                                mismatched.append(f"{scenario}/{delay}/{label}/{field}")
                            continue
                        largest = max(largest, abs(a - b))
                        if abs(a - b) > E4_PARITY_TOLERANCE:
                            mismatched.append(f"{scenario}/{delay}/{label}/{field}")
    return {"values_compared": compared, "largest_difference": largest, "mismatched": mismatched, "tolerance": E4_PARITY_TOLERANCE,
            "passes": compared > 0 and not mismatched}


# -- calibration and decision ------------------------------------------------------------


def calibrate(contract: Mapping[str, Any], report: Mapping[str, Any], bound: float) -> dict[str, Any]:
    """The largest grid rate at or below which every rate's visible gap is defined somewhere and within the bound everywhere it is defined."""

    spec = contract["simulation"]
    quantity = contract["calibration"]["quantity"]
    rates = spec["learning_rates"]
    out: dict[str, Any] = {}
    for delay in spec["outcome_delays"]:
        rows, star, open_ = [], None, True
        for k in sorted(range(len(rates)), key=lambda index: rates[index]):
            gaps = {s: report[problem_key("imitative", k, s, delay)]["rows"][quantity]["mean"] for s in spec["scenarios"]}
            defined = [value for value in gaps.values() if value is not None]
            allowed = bool(defined) and all(value <= bound for value in defined)
            rows.append({"rate_index": k, "rate": rates[k], "gaps": gaps, "allowed": allowed})
            if allowed and open_:
                star = k
            else:
                open_ = False
        out[f"delay_{delay}"] = {"rate_index": star, "rate": None if star is None else rates[star], "rows": rows}
    return out


def classify(row: Mapping[str, Any], material: float) -> str:
    low, high = row["mean_ci95"]
    if low >= material:
        return "MATERIAL"
    if high < material:
        return "BELOW_MATERIAL_DETECTED" if low > 0 else "BELOW_MATERIAL_NOT_DETECTED"
    return "UNRESOLVED"


def disposition(classes: Mapping[str, str] | None) -> str:
    if classes is None:
        return "NOT_CALIBRATED_THE_SMALLEST_RATE_EXCEEDS_THE_DATA_BOUND"
    material = [s for s in LOOP_SCENARIOS if classes[s] == "MATERIAL"]
    below = [s for s in LOOP_SCENARIOS if classes[s].startswith("BELOW_MATERIAL")]
    if len(material) == 2:
        return "THE_LOOP_HOLDS_AT_THE_DATA_ALLOWED_STRENGTH"
    if len(below) == 2:
        return "THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH"
    if material and below:
        return "THE_LOOP_HOLDS_ONLY_UNDER_GRANTS" if material == ["interdisciplinary_seed_grants"] else "THE_LOOP_HOLDS_ONLY_AT_BASELINE"
    return "UNRESOLVED_AT_THE_DATA_ALLOWED_STRENGTH"


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64))).astype(np.float64)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64))).astype(np.float64)
    return round(float(np.corrcoef(rx, ry)[0, 1]), 6)


def decide(contract: Mapping[str, Any], calibration: Mapping[str, Any], rows: Mapping[str, Any]) -> dict[str, Any]:
    spec, rule = contract["simulation"], contract["decision"]
    material = float(rule["material_failed_pivot_delta"])
    by_delay: dict[str, Any] = {}
    for delay in spec["outcome_delays"]:
        star = calibration["primary"][f"delay_{delay}"]["rate_index"]
        if star is None:
            classes, loop = None, None
        else:
            loop = {s: rows[f"imitative/{star}/{s}/delay_{delay}/failed_pivot_rate"] for s in LOOP_SCENARIOS}
            classes = {s: classify(loop[s], material) for s in LOOP_SCENARIOS}
        by_delay[f"delay_{delay}"] = {"rate_index": star, "rate": None if star is None else spec["learning_rates"][star],
                                      "classes": classes, "disposition": disposition(classes),
                                      "loop_rows": None if loop is None else {s: {k: loop[s][k] for k in ("mean", "mean_ci95")} for s in loop}}
    headline = by_delay[f"delay_{rule['primary_delay']}"]
    expectations = {}
    monotone = {}
    for delay in spec["outcome_delays"]:
        for scenario in ("interdisciplinary_seed_grants", "hot_field_hype"):
            pairs = [(r["rate"], r["gaps"][scenario]) for r in calibration["primary"][f"delay_{delay}"]["rows"] if r["gaps"][scenario] is not None]
            monotone[f"{scenario}/delay_{delay}"] = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
    expectations["X1"] = {"held": all(value is not None and value >= 0.9 for value in monotone.values()), "spearman": monotone}
    star_rate = headline["rate"]
    expectations["X2"] = {"held": star_rate is not None and star_rate <= 0.1875, "rate": star_rate}
    classes = headline["classes"]
    expectations["X3"] = {"held": classes is not None and classes["baseline"].startswith("BELOW_MATERIAL"),
                          "class": None if classes is None else classes["baseline"]}
    grants = None if headline["loop_rows"] is None else headline["loop_rows"]["interdisciplinary_seed_grants"]
    expectations["X4"] = {"held": grants is not None and grants["mean_ci95"][0] > 0, "row": grants}
    return {"status": f"O39_{headline['disposition']}", "disposition": headline["disposition"], "by_delay": by_delay,
            "expectations": expectations}


# -- run ---------------------------------------------------------------------------------


def run(contract: Mapping[str, Any], contract_path: Path, *, workers: int) -> tuple[dict[str, Any], dict[str, float]]:
    timing: dict[str, float] = {}
    started = time.monotonic()
    e4, e4_contract, study2 = load_modules(contract)
    tracked = json.loads(_binding(contract, TRACKED_ROLE).read_text(encoding="utf-8"))
    record = json.loads(_binding(contract, E4_RECORD_ROLE).read_text(encoding="utf-8"))
    results = run_simulation(contract, contract_path, workers=workers)
    timing["simulation_runs"] = round(time.monotonic() - started, 1)
    checks = simulation_checks(contract, study2, tracked, results, e4)
    require(checks["passes"], f"Study 2 parity failed: {checks}")

    mark = time.monotonic()
    problems, points, full_fits = estimation_problems(contract, e4, e4_contract, study2, results)
    draws = run_bootstrap(contract_path, e4, problems, e4._settings(e4_contract["estimator"]), workers=workers)
    timing["estimation"] = round(time.monotonic() - mark, 1)
    report = estimation_report(e4, problems, points, draws)
    parity = e4_parity(contract, record, report)
    rows = comparisons(contract, study2, results)
    calibration = {key: calibrate(contract, report, float(contract["calibration"][key]["value"])) for key in ("primary", "secondary")}
    valid = parity["passes"] and not full_fits["unconverged"]
    decision = decide(contract, calibration, rows) if valid else None
    summary = {
        "schema_version": "pivot_learning_strength_summary.v1",
        "status": decision["status"] if valid else "O39_INVALID_THE_E4_ESTIMATOR_IS_NOT_REPRODUCED",
        "decision": decision,
        "checks": {
            "study2_parity": checks,
            "e4_estimator_parity": parity,
            "full_sample_fits": full_fits,
            "bootstrap_unconverged_fits": sum(row["unconverged_fits"] for row in report.values()),
        },
        "calibration": calibration,
        "comparisons": rows,
        "arm_rates": arm_rates(contract, results),
        "estimates": report,
        "data_access": {"empirical_rows_read": 0, "source_of_the_data_bound": "THE_RECORDED_E4_RESULT"},
        "boundary": dict(contract["boundary"]),
    }
    timing["total"] = round(time.monotonic() - started, 1)
    return summary, timing


def write_outputs(output_dir: Path, summary: Mapping[str, Any], *, contract_path: Path, precommit: Mapping[str, Any],
                  timing: Mapping[str, float]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_exclusive(output_dir / "o39_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "pivot_learning_strength_receipt.v1",
        "status": "COMPLETE",
        "contract_sha256": _sha256(contract_path),
        "script_sha256": _sha256(Path(__file__)),
        "precommit": dict(precommit),
        "outputs": {"o39_summary.json": _sha256(output_dir / "o39_summary.json")},
    }
    _write_exclusive(output_dir / "o39_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write_exclusive(output_dir / "o39_timing.json", json.dumps(dict(timing), indent=2, sort_keys=True) + "\n")
    return receipt


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        load_modules(contract)
        tasks = simulation_tasks(contract)
        print(json.dumps({"status": "PRECOMMIT_PLAN_VALID", "contract_sha256": _sha256(config_path), "simulation_tasks": len(tasks),
                          "learning_rates": contract["simulation"]["learning_rates"]}, indent=2))
        return
    require(args.precommit_sha is not None and args.output_dir is not None, "run needs --precommit-sha and --output-dir")
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), f"run needs {', '.join(THREAD_VARIABLES)} set to 1")
    require(not args.output_dir.exists(), f"the output directory exists: {args.output_dir}")
    precommit = verify_precommit(args.precommit_sha, contract)
    summary, timing = run(contract, config_path, workers=args.workers)
    receipt = write_outputs(args.output_dir, summary, contract_path=config_path, precommit=precommit, timing=timing)
    print(json.dumps({"status": summary["status"], "receipt": receipt["outputs"], "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
