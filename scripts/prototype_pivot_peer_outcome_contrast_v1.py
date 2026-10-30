#!/usr/bin/env python3
"""Run the precommitted Pivot peer-outcome contrast v1 (recommended experiment 4).

Study 2 postulates that researchers learn from their neighbors' visible pivot
outcomes, imitatively or from failures only. This script estimates one
observable of that mechanism with one estimator, twice:

1. on Study 2's own arms, rerun unchanged through an environment slot and an
   agent slot, at the tracked outcome delay and at a three-step delay;
2. on the OpenAlex discovery partition, with coauthors as neighbors and the
   N11a v3 analyst label as the outcome.

The observable is a discrete-time entry hazard by the state of each
neighbor's outcome: visible and a success, visible and a failure, or not yet
visible with its eventual value. No confirmatory follow-up is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_peer_outcome_contrast_v1.yaml"
DEFAULT_AUTHORIZATION = ROOT / "experiments" / "configs" / "pivot_peer_outcome_contrast_authorization_v1.json"
SCHEMA = "pivot_peer_outcome_contrast.v1"
AUTHORIZATION_SCHEMA = "pivot_peer_outcome_contrast_authorization.v1"
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
STUDY2_ROLE = "STUDY2_SCRIPT_RUN_UNCHANGED_THROUGH_TWO_SLOTS"
STUDY1_ROLE = "STUDY1_SCRIPT_IMPORTED_BY_STUDY2"
TRACKED_ROLE = "STUDY2_TRACKED_REPLICATE_RUNS_FOR_PARITY"
N11B_BUILDER_ROLE = "N11B_BUILDER_REUSED_BY_HASH_BOUND_IMPORT"
N11B_CONTRACT_ROLE = "N11B_CONTRACT_S1_INPUTS_CLOCK_AND_G4_GUARD"
N11B_RESULT_ROLE = "N11B_RECORDED_RESULT_RECEIPT_HASH_AND_BYTE_IDENTITY"
N11A_RESULT_ROLE = "N11A_V3_RESULT_WHY_THIS_IS_NOT_THE_N11_LADDER"
IDENTITY = ["e4_summary.json", "e4_receipt.json", "private/cells.parquet"]
STATES = ("S", "F", "P_S", "P_F", "U")
STAGES = ("early", "mid", "senior", "bridge")
PARITY_KEYS = (
    "pivot_rate",
    "productive_pivot_rate",
    "failed_pivot_rate",
    "productive_share_of_pivots",
    "cumulative_failed_rate",
    "cumulative_productive_rate",
    "new_pivots_per_step",
    "new_failures_per_step",
)
CONTRASTS = {
    "outcome_gap_visible": {"S": 1.0, "F": -1.0},
    "outcome_gap_before_visible": {"P_S": 1.0, "P_F": -1.0},
    "visibility_did": {"S": 1.0, "F": -1.0, "P_S": -1.0, "P_F": 1.0},
    "success_following": {"S": 1.0, "P_S": -1.0},
    "failure_avoidance": {"F": 1.0, "P_F": -1.0},
}
MINIMUM_STATE_EVENTS = 10
CELL_SCHEMA = pa.schema([
    ("episode_id", pa.int16()), ("quarter", pa.int8()), ("prod_bin", pa.int8()), ("sub_bin", pa.int8()),
    ("age_bin", pa.int8()), ("incumbent", pa.int8()), ("n_bin", pa.int8()), ("lag_bin", pa.int8()),
    ("s", pa.int8()), ("f", pa.int8()), ("p_s", pa.int8()), ("p_f", pa.int8()), ("u", pa.int8()),
    ("weight_class", pa.string()), ("rows", pa.int64()), ("events", pa.int64()),
    ("rows_all", pa.int64()), ("events_all", pa.int64()),
])


class ContrastError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContrastError(message)


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


# -- contract and authorization ----------------------------------------------------------


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
    precondition = contract["precondition"]
    n11b = json.loads(_binding(contract, N11B_RESULT_ROLE).read_text(encoding="utf-8"))
    require(n11b["status"] == precondition["n11b_status"], "N11b did not pass")
    require(n11b["execution"]["receipt_sha256"] == contract["inputs"]["n11b_receipt_sha256"], "the N11b receipt differs from its record")
    require(n11b["execution"]["runs"]["development"] == contract["inputs"]["n11b_root"], "the N11b root differs from its record")
    require(all(n11b["execution"]["byte_identical"][name] for name in contract["inputs"]["n11b_files"]), "an N11b input was not reproduced")
    n11a = json.loads(_binding(contract, N11A_RESULT_ROLE).read_text(encoding="utf-8"))
    require(n11a["status"] == precondition["n11a_v3_status"], "the N11a v3 status differs")
    access = contract["data_access"]
    require(access["partition"] == "DISCOVERY_ONLY", "the study may read the discovery partition only")
    require(contract["authorization"]["confirmatory_outcomes"] == "NOT_AUTHORIZED", "confirmatory outcomes may not be authorized here")
    require(contract["authorization"]["outcomes"] == "DISCOVERY_PARTITION_ONLY", "outcomes are authorized for discovery only")
    simulation = contract["simulation"]
    require(simulation["outcome_delays"] == [0, 3], "the outcome delays are 0 and 3")
    require(simulation["arms"][0] == "frozen", "the frozen arm comes first")
    estimator = contract["estimator"]
    require(estimator["state_model"]["reference"] == "P_F", "the state reference is P_F")
    require(int(estimator["bootstrap"]["draws"]) > 0, "the bootstrap needs draws")
    require(contract["execution"]["identity_artifacts"] == IDENTITY, "identity artifacts differ")
    require(contract["boundary"]["identifies_learning"] is False, "the study does not identify learning")


def load_authorization(path: Path, contract_path: Path, script_path: Path) -> dict[str, Any]:
    authorization = json.loads(Path(path).read_text(encoding="utf-8"))
    require(authorization.get("schema_version") == AUTHORIZATION_SCHEMA, "unexpected authorization schema")
    require(authorization.get("status") == "AUTHORIZED_DISCOVERY_PEER_OUTCOME_CONTRAST", "the run is not authorized")
    require(authorization.get("contract_sha256") == _sha256(contract_path), "the authorization names another contract")
    require(authorization.get("script_sha256") == _sha256(script_path), "the authorization names another script")
    require(bool(authorization.get("human_authorization", {}).get("user_instruction")), "the authorization must quote the user")
    require(authorization.get("outcomes") == "DISCOVERY_PARTITION_ONLY", "outcomes: discovery only")
    for key in ("confirmatory_outcomes", "g4_outcomes"):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"{key} must stay unauthorized")
    return authorization


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


def load_study2(contract: Mapping[str, Any]) -> ModuleType:
    """Study 2 by its bound path; it imports Study 1 from examples/, which must be the bound file."""

    study2_path = _binding(contract, STUDY2_ROLE)
    examples = str(study2_path.parent)
    if examples not in sys.path:
        sys.path.insert(0, examples)
    name = "research_pivot_learning_study_for_e4"
    if name in sys.modules:
        module = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, study2_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    require(Path(module.study1.__file__).resolve() == _binding(contract, STUDY1_ROLE).resolve(), "Study 2 imported another Study 1")
    return module


def load_n11b(contract: Mapping[str, Any]) -> ModuleType:
    name = "n11b_builder_for_e4"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _binding(contract, N11B_BUILDER_ROLE))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# -- Study 2 through two slots -----------------------------------------------------------


_ENVIRONMENT_CLASSES: dict[int, type] = {}


def delayed_environment_class(study2: ModuleType) -> type:
    """Study 2's environment with an event history; agents see the events of step s - 1 - delay."""

    key = id(study2)
    if key not in _ENVIRONMENT_CLASSES:

        @dataclass
        class DelayedEnvironment(study2.PivotEnvironment):
            delay: int = 0
            history: list = field(default_factory=list)

            def advance(self) -> None:
                self.history.append(self.new_events)
                self.step += 1
                index = len(self.history) - 1 - self.delay
                self.last_events = self.history[index] if index >= 0 else []
                self.new_events = []

        _ENVIRONMENT_CLASSES[key] = DelayedEnvironment
    return _ENVIRONMENT_CLASSES[key]


def run_logged_arm(study2: ModuleType, *, scenario: Any, config: Any, replicate: int, mode: str, delay: int) -> tuple[dict, dict]:
    """``study2.run_arm`` unchanged, with the environment and agent classes swapped for logged ones."""

    environment_class = delayed_environment_class(study2)
    agents: list[Any] = []
    environments: list[Any] = []
    base_agent = study2.PivotPolicyAgent

    class LoggedAgent(base_agent):
        def __post_init__(self) -> None:
            super().__post_init__()
            agents.append(self)

    def make_environment(**kwargs: Any) -> Any:
        environment = environment_class(**kwargs, delay=delay)
        environments.append(environment)
        return environment

    saved = (study2.PivotEnvironment, study2.PivotPolicyAgent)
    study2.PivotEnvironment, study2.PivotPolicyAgent = make_environment, LoggedAgent
    try:
        run = study2.run_arm(scenario=scenario, config=config, replicate=replicate, learning_mode=mode)
    finally:
        study2.PivotEnvironment, study2.PivotPolicyAgent = saved
    require(len(environments) == 1 and len(agents) == config.agent_count, "the slots did not capture one run")
    agents.sort(key=lambda agent: agent.agent_id)
    log = {
        "stage": [STAGES.index(agent.base.career_stage) for agent in agents],
        "pivot_step": [-1 if agent.pivot_step is None else int(agent.pivot_step) for agent in agents],
        "productive": [bool(agent.base.productive_pivot) for agent in agents],
        "neighbors": [list(map(int, row)) for row in environments[0].neighbors],
    }
    return run, log


def run_simulation(contract: Mapping[str, Any], study2: ModuleType, tracked: Mapping[str, Any]) -> dict[str, Any]:
    simulation = contract["simulation"]
    config = study2.LearningStudyConfig()
    require({k: v for k, v in tracked["config"].items()} == {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()},
            "the Study 2 config differs from the tracked one")
    scenarios = {s.name: s for s in study2.study1.study_scenarios()}
    logs: dict[tuple[str, str, int], list[dict]] = {}
    parity = {"delay_0_runs_checked": 0, "delay_0_runs_equal": 0, "frozen_delay_3_runs_equal": 0}
    for scenario_name in simulation["scenarios"]:
        for delay in simulation["outcome_delays"]:
            for mode in simulation["arms"]:
                rows = []
                for replicate in range(int(simulation["replicates"])):
                    run, log = run_logged_arm(study2, scenario=scenarios[scenario_name], config=config, replicate=replicate, mode=mode, delay=delay)
                    if delay == 0:
                        recorded = tracked["scenarios"][scenario_name][mode]["replicate_runs"][replicate]
                        parity["delay_0_runs_checked"] += 1
                        parity["delay_0_runs_equal"] += int(all(run[key] == recorded[key] for key in PARITY_KEYS))
                    elif mode == "frozen":
                        reference = logs[(scenario_name, mode, 0)][replicate]
                        parity["frozen_delay_3_runs_equal"] += int(log == reference)
                    rows.append(log)
                logs[(scenario_name, mode, delay)] = rows
    replicates = int(simulation["replicates"])
    parity["passes"] = (
        parity["delay_0_runs_checked"] == parity["delay_0_runs_equal"] == len(simulation["scenarios"]) * len(simulation["arms"]) * replicates
        and parity["frozen_delay_3_runs_equal"] == len(simulation["scenarios"]) * replicates
    )
    return {"logs": logs, "parity": parity, "burn_in": int(config.burn_in_steps), "steps": int(config.steps)}


def simulation_rows(logs: Sequence[Mapping[str, Any]], *, delay: int, burn_in: int, steps: int) -> dict[str, np.ndarray]:
    """Person-step rows of one scenario, arm, and delay over every replicate."""

    columns: dict[str, list[int]] = {key: [] for key in ("cluster", "time", "stage", "event", "n_visible", "lag", *STATES)}
    entry = {"cluster": [], "time": [], "label": []}
    first, last = burn_in + 1, steps
    for replicate, log in enumerate(logs):
        pivot, productive, neighbors, stage = log["pivot_step"], log["productive"], log["neighbors"], log["stage"]
        for agent, step in enumerate(pivot):
            if step >= first:
                entry["cluster"].append(replicate)
                entry["time"].append(step - first)
                entry["label"].append(int(productive[agent]))
        for agent in range(len(pivot)):
            for r in range(first, last + 1):
                if 0 <= pivot[agent] < r:
                    break
                visible = [j for j in neighbors[agent] if 0 <= pivot[j] < r]
                states = dict.fromkeys(STATES, 0)
                for j in visible:
                    shown = r >= pivot[j] + 1 + delay
                    states[("S" if productive[j] else "F") if shown else ("P_S" if productive[j] else "P_F")] = 1
                columns["cluster"].append(replicate)
                columns["time"].append(r - first)
                columns["stage"].append(stage[agent])
                columns["event"].append(int(pivot[agent] == r))
                columns["n_visible"].append(len(visible))
                columns["lag"].append(r - min(pivot[j] + 1 for j in visible) if visible else -1)
                for state in STATES:
                    columns[state].append(states[state])
    rows = {key: np.asarray(value, dtype=np.int64) for key, value in columns.items()}
    rows["entry"] = {key: np.asarray(value, dtype=np.int64) for key, value in entry.items()}
    return rows


# -- binning and cells -------------------------------------------------------------------


def bin_index(values: np.ndarray, edges: Sequence[Sequence[int | None]]) -> np.ndarray:
    """The index of the [low, high] bin each value falls in; None is unbounded."""

    values = np.asarray(values)
    out = np.full(values.shape, -1, dtype=np.int64)
    for index, (low, high) in enumerate(edges):
        inside = np.ones(values.shape, dtype=bool)
        if low is not None:
            inside &= values >= low
        if high is not None:
            inside &= values <= high
        out[(out < 0) & inside] = index
    require(bool((out >= 0).all()), "a value falls outside every registered bin")
    return out


def _one_hot(index: np.ndarray, levels: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    columns = [(index == level).astype(np.float64) for level in range(1, levels)]
    return (np.column_stack(columns) if columns else np.zeros((index.size, 0))), [f"{prefix}{level}" for level in range(1, levels)]


@dataclass
class Cells:
    """Rows collapsed to identical covariates: fixed-effect group, cluster, trials, events, design."""

    group: np.ndarray
    cluster: np.ndarray
    trials: np.ndarray
    events: np.ndarray
    design: np.ndarray
    names: list[str]
    state_events: dict[str, float] = field(default_factory=dict)


def collapse(group: np.ndarray, cluster: np.ndarray, design: np.ndarray, trials: np.ndarray, events: np.ndarray, names: list[str],
             state_events: Mapping[str, float] | None = None) -> Cells:
    keys = np.column_stack([group, cluster, design]).astype(np.float64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return Cells(
        group=unique[:, 0].astype(np.int64),
        cluster=unique[:, 1].astype(np.int64),
        trials=np.bincount(inverse, weights=trials, minlength=unique.shape[0]),
        events=np.bincount(inverse, weights=events, minlength=unique.shape[0]),
        design=unique[:, 2:],
        names=list(names),
        state_events=dict(state_events or {}),
    )


def _state_design(states: Mapping[str, np.ndarray], n_bin: np.ndarray, lag_bin: np.ndarray, estimator: Mapping[str, Any]) -> tuple[np.ndarray, list[str], str]:
    present = [s for s in STATES if np.asarray(states[s]).any()]
    reference = "P_F" if "P_F" in present else "F"
    terms = [s for s in STATES if s in present and s != reference]
    model = estimator["state_model"]
    n_cols, n_names = _one_hot(n_bin, len(model["controls"]["visible_coauthor_entries"]), "n")
    lag_cols, lag_names = _one_hot(lag_bin, len(model["controls"]["quarters_since_the_first_visible_entry"]), "lag")
    design = np.column_stack([*(np.asarray(states[s], dtype=np.float64) for s in terms), n_cols, lag_cols])
    return design, [*terms, *n_names, *lag_names], reference


def simulation_cells(rows: Mapping[str, Any], estimator: Mapping[str, Any]) -> dict[str, Cells]:
    group = rows["cluster"] * 12 + rows["time"]
    stage_cols, stage_names = _one_hot(rows["stage"], len(STAGES), "stage")
    ones = np.ones(rows["event"].size)
    exposed = (rows["n_visible"] > 0).astype(np.float64)
    exposure = collapse(group, rows["cluster"], np.column_stack([exposed, stage_cols]), ones, rows["event"].astype(np.float64),
                        ["exposed", *stage_names])
    keep = rows["n_visible"] > 0
    model = estimator["state_model"]["controls"]
    n_bin = bin_index(rows["n_visible"][keep], model["visible_coauthor_entries"])
    lag_bin = bin_index(rows["lag"][keep], model["quarters_since_the_first_visible_entry"])
    states = {s: rows[s][keep] for s in STATES}
    design, names, reference = _state_design(states, n_bin, lag_bin, estimator)
    events = rows["event"][keep].astype(np.float64)
    state_events = {s: float(events[states[s] == 1].sum()) for s in STATES}
    state_events["reference"] = reference
    state = collapse(group[keep], rows["cluster"][keep], np.column_stack([design, stage_cols[keep]]), ones[keep], events,
                     [*names, *stage_names], state_events)
    entry = rows["entry"]
    entry_cells = collapse(entry["cluster"], entry["cluster"], (entry["time"] / 11.0)[:, None], np.ones(entry["label"].size),
                           entry["label"].astype(np.float64), ["entry_time"])
    return {"exposure": exposure, "state": state, "entry_order": entry_cells}


# -- the estimator -----------------------------------------------------------------------


def _log1pexp(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def fit_logit(cells: Cells, *, ridge: float, tolerance: float, max_iterations: int, max_step: float, weights: np.ndarray | None = None,
              beta_start: np.ndarray | None = None) -> dict[str, Any]:
    """Weighted logistic MLE with unpenalized group effects and a ridge on the design coefficients.

    Groups with no events, or with only events, are dropped: their effect
    diverges and they carry no information on the coefficients. Newton steps
    eliminate the group block. A step is capped at ``max_step`` on the log-odds
    scale, and a step that does not raise the penalized likelihood is never
    taken: the fit then stops unconverged.
    """

    scale = np.ones(cells.group.size) if weights is None else np.asarray(weights, dtype=np.float64)
    trials, events = cells.trials * scale, cells.events * scale
    present = trials > 0
    size_groups = int(cells.group.max()) + 1
    group_events = np.bincount(cells.group[present], weights=events[present], minlength=size_groups)
    group_all = np.bincount(cells.group[present], weights=trials[present], minlength=size_groups)
    keep = present & (group_events[cells.group] > 0) & (group_events[cells.group] < group_all[cells.group])
    _, g = np.unique(cells.group[keep], return_inverse=True)
    g = g.reshape(-1)
    n, y, x = trials[keep], events[keep], cells.design[keep]
    count, p = int(g.max()) + 1 if g.size else 0, x.shape[1]
    if count == 0:
        return {"beta": np.full(p, np.nan), "converged": False, "iterations": 0, "groups": 0}
    group_trials = np.bincount(g, weights=n, minlength=count)
    group_rate = np.clip(np.bincount(g, weights=y, minlength=count) / group_trials, 1e-12, 1 - 1e-12)
    alpha = np.log(group_rate) - np.log1p(-group_rate)
    beta = np.zeros(p) if beta_start is None or not np.all(np.isfinite(beta_start)) else np.asarray(beta_start, dtype=np.float64).copy()

    def loglik(a: np.ndarray, b: np.ndarray) -> float:
        eta = a[g] + x @ b
        return float(np.sum(y * eta - n * _log1pexp(eta)) - 0.5 * ridge * float(b @ b))

    current = loglik(alpha, beta)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        eta = alpha[g] + x @ beta
        mu = np.exp(-np.logaddexp(0.0, -eta))
        w = n * mu * (1.0 - mu)
        resid = y - n * mu
        grad_a = np.bincount(g, weights=resid, minlength=count)
        grad_b = x.T @ resid - ridge * beta
        d = np.maximum(np.bincount(g, weights=w, minlength=count), 1e-300)
        cross = np.column_stack([np.bincount(g, weights=w * x[:, k], minlength=count) for k in range(p)]) if p else np.zeros((count, 0))
        hess_b = (x * w[:, None]).T @ x + ridge * np.eye(p)
        schur = hess_b - cross.T @ (cross / d[:, None])
        step_b = np.linalg.solve(schur, grad_b - cross.T @ (grad_a / d)) if p else np.zeros(0)
        step_a = (grad_a - cross @ step_b) / d
        raw = max(float(np.max(np.abs(step_a))), float(np.max(np.abs(step_b))) if p else 0.0)
        size = min(1.0, max_step / raw) if raw > 0 else 1.0
        improved = False
        while size >= 1e-12:
            candidate_a, candidate_b = alpha + size * step_a, beta + size * step_b
            value = loglik(candidate_a, candidate_b)
            if value >= current - 1e-12 * max(1.0, abs(current)):
                improved = True
                break
            size *= 0.5
        if not improved:
            break
        alpha, beta, current = candidate_a, candidate_b, value
        largest = max(float(np.max(np.abs(size * step_a))), float(np.max(np.abs(size * step_b))) if p else 0.0)
        if largest < tolerance:
            converged = True
            break
    return {"beta": beta, "converged": converged, "iterations": iterations, "groups": count}


def contrast_values(beta: np.ndarray, names: Sequence[str], reference: str) -> dict[str, float | None]:
    index = {name: k for k, name in enumerate(names)}
    out: dict[str, float | None] = {}
    for key, weights in CONTRASTS.items():
        total = 0.0
        defined = True
        for state, weight in weights.items():
            if state == reference:
                continue
            if state not in index:
                defined = False
                break
            total += weight * float(beta[index[state]])
        out[key] = total if defined else None
    return out


def summarize_draws(point: float | None, draws: np.ndarray) -> dict[str, Any]:
    finite = draws[np.isfinite(draws)]
    if point is None or finite.size == 0:
        return {"mean": None, "low": None, "high": None, "finite_draws": int(finite.size)}
    low, high = np.percentile(finite, [2.5, 97.5])
    return {"mean": round(float(point), 6), "low": round(float(low), 6), "high": round(float(high), 6), "finite_draws": int(finite.size)}


# -- bootstrap ---------------------------------------------------------------------------


_WORKER: dict[str, Any] = {}


def _init_worker(problems: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    _WORKER["problems"] = problems
    _WORKER["settings"] = settings


def _problem_values(problem: Mapping[str, Any], multiplicity: np.ndarray, settings: Mapping[str, Any]) -> tuple[list[float], int]:
    """The problem's tracked quantities on one draw, and how many of its fits did not converge."""

    values: list[float] = []
    unconverged = 0
    for model in problem["models"]:
        cells: Cells = model["cells"]
        fit = fit_logit(cells, **settings, weights=multiplicity[cells.cluster], beta_start=model["beta"])
        if not fit["converged"]:
            unconverged += 1
            values.extend([np.nan] * len(model["quantities"]))
            continue
        contrasts = contrast_values(fit["beta"], cells.names, model["reference"])
        for quantity in model["quantities"]:
            if quantity.startswith("beta:"):
                values.append(float(fit["beta"][cells.names.index(quantity[5:])]))
            else:
                value = contrasts[quantity]
                values.append(np.nan if value is None else value)
    return values, unconverged


def _run_chunk(task: tuple[str, int, int]) -> tuple[str, int, list[list[float]], int]:
    key, start, stop = task
    problem = _WORKER["problems"][key]
    settings = _WORKER["settings"]
    out, failures = [], 0
    for draw in range(start, stop):
        values, unconverged = _problem_values(problem, problem["multiplicities"][draw], settings)
        failures += unconverged
        out.append(values)
    return key, start, out, failures


def bootstrap(problems: Mapping[str, Any], settings: Mapping[str, Any], *, workers: int, chunk: int = 50) -> dict[str, dict[str, Any]]:
    tasks = []
    for key, problem in problems.items():
        draws = problem["multiplicities"].shape[0]
        tasks.extend((key, start, min(start + chunk, draws)) for start in range(0, draws, chunk))
    results: dict[str, list[tuple[int, list[list[float]]]]] = {key: [] for key in problems}
    failures = dict.fromkeys(problems, 0)
    if workers <= 1:
        _init_worker(problems, settings)
        outputs = [_run_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
                                 initargs=(problems, settings)) as pool:
            outputs = list(pool.map(_run_chunk, tasks))
    for key, start, values, failed in outputs:
        results[key].append((start, values))
        failures[key] += failed
    return {
        key: {"draws": np.asarray([v for _start, chunk_values in sorted(results[key], key=lambda item: item[0]) for v in chunk_values], dtype=np.float64),
              "unconverged_fits": failures[key]}
        for key in problems
    }


def multiplicities(clusters: int, draws: int, seed: int, *key: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence((seed, *key)))
    return rng.multinomial(clusters, np.full(clusters, 1.0 / clusters), size=draws).astype(np.float64)


# -- simulation estimates ----------------------------------------------------------------


def simulation_problems(contract: Mapping[str, Any], simulation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    estimator = contract["estimator"]
    settings = _settings(estimator)
    spec = contract["simulation"]
    replicates = int(spec["replicates"])
    seed = int(estimator["bootstrap"]["seed"])
    draws = int(estimator["bootstrap"]["draws"])
    problems: dict[str, Any] = {}
    points: dict[str, Any] = {}
    for s_index, scenario in enumerate(spec["scenarios"]):
        for delay in spec["outcome_delays"]:
            models = []
            for arm in spec["arms"]:
                rows = simulation_rows(simulation["logs"][(scenario, arm, delay)], delay=delay, burn_in=simulation["burn_in"], steps=simulation["steps"])
                cells = simulation_cells(rows, estimator)
                for model_key, quantities in (("exposure", ["beta:exposed"]), ("state", list(CONTRASTS)), ("entry_order", ["beta:entry_time"])):
                    model_cells = cells[model_key]
                    fit = fit_logit(model_cells, **settings)
                    require(fit["converged"], f"a full-sample simulation fit did not converge: {scenario} {arm} {delay} {model_key}")
                    reference = model_cells.state_events.get("reference", "P_F")
                    models.append({"arm": arm, "model": model_key, "cells": model_cells, "beta": fit["beta"], "reference": reference,
                                   "quantities": quantities})
            key = f"simulation/{scenario}/delay_{delay}"
            problems[key] = {"models": models, "multiplicities": multiplicities(replicates, draws, seed, 1, s_index, delay)}
            points[key] = [_point_values(m) for m in models]
    return problems, points


def _settings(estimator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ridge": float(estimator["ridge_on_non_fixed_effect_coefficients"]),
        "tolerance": float(estimator["newton"]["tolerance"]),
        "max_iterations": int(estimator["newton"]["max_iterations"]),
        "max_step": float(estimator["newton"]["max_step_log_odds"]),
    }


def _point_values(model: Mapping[str, Any]) -> list[float | None]:
    cells: Cells = model["cells"]
    contrasts = contrast_values(model["beta"], cells.names, model["reference"])
    out: list[float | None] = []
    for quantity in model["quantities"]:
        if quantity.startswith("beta:"):
            out.append(float(model["beta"][cells.names.index(quantity[5:])]))
        else:
            out.append(contrasts[quantity])
    return out


def _quantity_defined(model: Mapping[str, Any], quantity: str) -> bool:
    if quantity.startswith("beta:"):
        return True
    events = model["cells"].state_events
    reference = model["reference"]
    states = list(CONTRASTS[quantity])
    if any(state not in model["cells"].names and state != reference for state in states):
        return False
    return all(events.get(state, 0.0) >= MINIMUM_STATE_EVENTS for state in states)


def simulation_report(problems: Mapping[str, Any], points: Mapping[str, Any], draws: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key, problem in problems.items():
        values = draws[key]["draws"]
        column = 0
        by_arm: dict[str, dict[str, Any]] = {}
        index: dict[tuple[str, str], int] = {}
        for model, point in zip(problem["models"], points[key], strict=True):
            for quantity, value in zip(model["quantities"], point, strict=True):
                name = quantity[5:] if quantity.startswith("beta:") else quantity
                label = f"{model['model']}/{name}"
                defined = _quantity_defined(model, quantity)
                by_arm.setdefault(model["arm"], {})[label] = summarize_draws(value if defined else None, values[:, column])
                index[(model["arm"], label)] = column
                column += 1
        attributable: dict[str, dict[str, Any]] = {}
        for arm in by_arm:
            if arm == "frozen":
                continue
            for label in by_arm[arm]:
                own, base = by_arm[arm][label], by_arm["frozen"][label]
                if own["mean"] is None or base["mean"] is None:
                    attributable.setdefault(arm, {})[label] = summarize_draws(None, np.array([]))
                    continue
                diff = values[:, index[(arm, label)]] - values[:, index[("frozen", label)]]
                attributable.setdefault(arm, {})[label] = summarize_draws(own["mean"] - base["mean"], diff)
        state_events = {model["arm"]: {k: v for k, v in model["cells"].state_events.items()} for model in problem["models"] if model["model"] == "state"}
        report[key] = {"by_arm": by_arm, "learning_attributable": attributable, "state_events": state_events,
                       "unconverged_fits": draws[key]["unconverged_fits"]}
    return report


# -- the empirical build -----------------------------------------------------------------


def _risk_condition(alias: str, author: str) -> str:
    """The N11b risk rule for an author in an episode-year row ``alias`` with y, field, topic."""

    return (
        f"EXISTS (SELECT 1 FROM field_years fy WHERE fy.author_id = {author} AND fy.field = {alias}.field "
        f"AND fy.year BETWEEN {alias}.y - 3 AND {alias}.y - 1) "
        f"AND NOT EXISTS (SELECT 1 FROM ft p WHERE p.author_id = {author} AND p.topic = {alias}.topic AND year(p.first_date) < {alias}.y) "
        f"AND NOT EXISTS (SELECT 1 FROM guard gd WHERE gd.author_id = {author})"
    )


def build_empirical(inputs: Mapping[str, Path], guard_path: Path, guard_column: str, n11b_private: Mapping[str, Path], n11b: ModuleType,
                    estimator: Mapping[str, Any], *, threads: int = 8, structure_only: bool = False) -> dict[str, Any]:
    """Discovery-episode cells for the exposure, state, and entry-order models, with parity counts.

    Follow-up topic works are read only for S1 entrants of discovery episodes,
    and only inside their label window. ``structure_only`` stops after the
    N11b parity, before any label, entry work, or follow-up is read.
    """

    import duckdb

    off, precise, visible = n11b._offset, n11b._precise, n11b._visible_from
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute("CREATE TABLE w AS SELECT work_id, year, date, topic, subfield, field FROM read_parquet(?)", [str(inputs["works"])])
    con.execute("CREATE TABLE a AS SELECT work_id, author_id FROM read_parquet(?)", [str(inputs["authorships"])])
    con.execute("CREATE TABLE ep AS SELECT episode_id, field, topic, subfield, t FROM read_parquet(?) WHERE partition = 'discovery'", [str(inputs["episodes"])])
    con.execute(
        'CREATE TABLE s1e AS SELECT s.episode_id, s.author_id, s.entry_date, s."offset" AS entry_offset, s.precise '
        "FROM read_parquet(?) s JOIN ep USING (episode_id)", [str(inputs["entrants"])])
    con.execute(f'CREATE TABLE guard AS SELECT DISTINCT "{guard_column}" AS author_id FROM read_parquet(?)', [str(guard_path)])
    con.execute("CREATE TABLE aw AS SELECT a.author_id, w.work_id, w.topic, w.subfield, w.field, w.year, w.date FROM a JOIN w USING (work_id)")
    con.execute("CREATE TABLE first_any AS SELECT author_id, MIN(year) AS first_year FROM aw GROUP BY author_id")
    con.execute("CREATE TABLE field_years AS SELECT DISTINCT author_id, field, year FROM aw")
    con.execute("CREATE TABLE ft AS SELECT author_id, topic, MIN(date) AS first_date FROM aw WHERE topic IN (SELECT topic FROM ep) GROUP BY author_id, topic")
    con.execute("CREATE TABLE ey AS SELECT episode_id, field, topic, subfield, t, k AS ry, t + k AS y FROM ep, range(3) AS r(k)")

    # Coauthor entrants: anyone whose first topic work falls in the response years, visible by the N11b rule.
    con.execute(
        f"""
        CREATE TABLE cent AS
        SELECT ep.episode_id, ft.author_id AS j, ft.first_date, {visible("ft.first_date", "ep.t")} AS v
        FROM ft JOIN ep ON ft.topic = ep.topic WHERE year(ft.first_date) BETWEEN ep.t AND ep.t + 2
        """
    )
    # Dyads: an at-risk author and a coauthor entrant who share an extract work dated y-3..y-1.
    con.execute(
        """
        CREATE TABLE dyad_raw AS
        SELECT DISTINCT c.episode_id, e.ry, a2.author_id AS i, c.j
        FROM cent c JOIN ey e USING (episode_id)
        JOIN aw ON aw.author_id = c.j AND aw.year BETWEEN e.y - 3 AND e.y - 1
        JOIN a a2 ON a2.work_id = aw.work_id AND a2.author_id <> c.j
        """
    )
    con.execute(f"CREATE TABLE dyad AS SELECT d.* FROM dyad_raw d JOIN ey e USING (episode_id, ry) WHERE {_risk_condition('e', 'd.i')}")
    con.execute("CREATE TABLE xa AS SELECT DISTINCT episode_id, i AS author_id FROM dyad")

    # Every at-risk year of every exposed author, with N11b's rows, events, and features.
    con.execute(
        f"""
        CREATE TABLE xrisk AS
        SELECT x.episode_id, x.author_id, e.ry, e.y, e.t, e.field, e.subfield, e.topic, f.first_date
        FROM xa x JOIN ey e USING (episode_id)
        LEFT JOIN ft f ON f.author_id = x.author_id AND f.topic = e.topic
        WHERE {_risk_condition('e', 'x.author_id')}
        """
    )
    con.execute(
        f"""
        CREATE TABLE xsy AS
        SELECT *, COALESCE(year(first_date) = y, FALSE) AS entry_year, {off("first_date", "t")} AS entry_offset,
               COALESCE({precise("first_date")}, FALSE) AS is_precise
        FROM xrisk
        """
    )
    con.execute(
        """
        CREATE TABLE xrows AS
        SELECT s.episode_id, s.author_id, s.ry, q.q AS quarter,
               CASE WHEN s.entry_year AND s.is_precise AND q.q = s.entry_offset THEN 1
                    WHEN s.entry_year AND NOT s.is_precise THEN 2 ELSE 0 END AS event
        FROM xsy s, range(12) AS q(q)
        WHERE q.q // 4 = s.ry AND NOT (s.entry_year AND s.is_precise AND q.q > s.entry_offset)
        """
    )
    con.execute(
        """
        CREATE TABLE xwin AS
        SELECT k.episode_id, k.author_id, k.ry, aw.work_id, aw.field = k.field AS in_field,
               aw.field = k.field AND aw.subfield = k.subfield AS in_subfield
        FROM (SELECT DISTINCT episode_id, author_id, ry, y, field, subfield FROM xsy) k
        JOIN aw ON aw.author_id = k.author_id AND aw.year BETWEEN k.y - 3 AND k.y - 1
        """
    )
    con.execute(
        """
        CREATE TABLE xfeat AS
        SELECT k.episode_id, k.author_id, k.ry,
               COALESCE(p.productivity_works, 0) AS productivity_works, COALESCE(p.subfield_works, 0) AS subfield_works,
               k.y - fa.first_year AS career_age_years, COALESCE(c.incumbent_coauthors, 0) AS incumbent_coauthors
        FROM (SELECT DISTINCT episode_id, author_id, ry, y, t, topic FROM xsy) k
        JOIN first_any fa USING (author_id)
        LEFT JOIN (
            SELECT episode_id, author_id, ry,
                   COUNT(DISTINCT work_id) FILTER (WHERE in_field) AS productivity_works,
                   COUNT(DISTINCT work_id) FILTER (WHERE in_subfield) AS subfield_works
            FROM xwin GROUP BY episode_id, author_id, ry
        ) p USING (episode_id, author_id, ry)
        LEFT JOIN (
            SELECT wn.episode_id, wn.author_id, wn.ry, COUNT(DISTINCT co.author_id) AS incumbent_coauthors
            FROM xwin wn JOIN ey e USING (episode_id, ry)
            JOIN a co ON co.work_id = wn.work_id AND co.author_id <> wn.author_id
            JOIN ft f ON f.author_id = co.author_id AND f.topic = e.topic
            WHERE year(f.first_date) < e.t
            GROUP BY wn.episode_id, wn.author_id, wn.ry
        ) c USING (episode_id, author_id, ry)
        """
    )

    # Structural parity with N11b before any label is read: rows, events, features, coauthor entry counts.
    con.execute(
        """
        CREATE TABLE xvis AS
        SELECT r.episode_id, r.author_id, r.ry, r.quarter, r.event, COUNT(c.j) AS n_visible, MIN(c.v) AS first_v
        FROM xrows r
        LEFT JOIN dyad d ON d.episode_id = r.episode_id AND d.i = r.author_id AND d.ry = r.ry
        LEFT JOIN cent c ON c.episode_id = d.episode_id AND c.j = d.j AND c.v <= r.quarter
        GROUP BY r.episode_id, r.author_id, r.ry, r.quarter, r.event
        """
    )
    con.execute("CREATE TABLE np AS SELECT p.* FROM read_parquet(?) p JOIN ep USING (episode_id)", [str(n11b_private["panel"])])
    con.execute("CREATE TABLE samp AS SELECT episode_id, sampling_probability FROM read_parquet(?) WHERE partition = 'discovery'",
                [str(n11b_private["episodes_sampling"])])
    parity = {
        "exposed_panel_rows_missing": _scalar(con, """
            SELECT COUNT(*) FROM np JOIN xa USING (episode_id, author_id)
            ANTI JOIN xvis x ON x.episode_id = np.episode_id AND x.author_id = np.author_id AND x.quarter = np.quarter"""),
        "exposed_panel_rows_extra": _scalar(con, """
            SELECT COUNT(*) FROM xvis x JOIN (SELECT DISTINCT episode_id, author_id FROM np) k USING (episode_id, author_id)
            ANTI JOIN np ON np.episode_id = x.episode_id AND np.author_id = x.author_id AND np.quarter = x.quarter"""),
        "exposed_panel_rows_differing": _scalar(con, """
            SELECT COUNT(*) FROM np JOIN xvis x USING (episode_id, author_id, quarter) JOIN xfeat f USING (episode_id, author_id, ry)
            WHERE np.event <> x.event OR np.coauthor_entries_visible <> x.n_visible OR np.productivity_works <> f.productivity_works
               OR np.subfield_works <> f.subfield_works OR np.career_age_years <> f.career_age_years
               OR np.incumbent_coauthors <> f.incumbent_coauthors OR np.response_year <> x.ry"""),
        "unexposed_panel_rows_with_a_visible_entry": _scalar(con, """
            SELECT COUNT(*) FROM np ANTI JOIN xa USING (episode_id, author_id) WHERE np.coauthor_entries_visible > 0"""),
        "discovery_episodes_without_a_probability": _scalar(con, "SELECT COUNT(*) FROM ep ANTI JOIN samp USING (episode_id)"),
        "s1_entrants_missing_from_their_rows": _scalar(con, """
            SELECT COUNT(*) FROM s1e ANTI JOIN (SELECT episode_id, author_id FROM np UNION SELECT episode_id, author_id FROM xvis) k
            USING (episode_id, author_id)"""),
    }
    if any(parity.values()) or structure_only:
        structure = {
            "dyads": _scalar(con, "SELECT COUNT(*) FROM dyad"),
            "exposed_authors": _scalar(con, "SELECT COUNT(*) FROM xa"),
            "exposed_rows": _scalar(con, "SELECT COUNT(*) FROM xvis"),
            "exposed_rows_with_a_visible_entry": _scalar(con, "SELECT COUNT(*) FROM xvis WHERE n_visible > 0"),
        }
        con.close()
        return {"stopped": bool(any(parity.values())), "parity": dict(sorted(parity.items())), "structure": structure}

    # Labels of discovery S1 entrants: the N11a v3 analyst rule, inside the label window only.
    con.execute(
        f"""
        CREATE TABLE follow AS
        SELECT DISTINCT s.episode_id, s.author_id
        FROM s1e s JOIN ep USING (episode_id)
        JOIN aw f ON f.author_id = s.author_id AND f.topic = ep.topic
        WHERE (s.precise AND f.year BETWEEN year(s.entry_date) AND year(s.entry_date) + 2 AND (
                   ({precise("f.date")} AND {off("f.date", "ep.t")} BETWEEN s.entry_offset + 1 AND s.entry_offset + 4)
                OR (NOT {precise("f.date")} AND 4 * (f.year - ep.t) >= s.entry_offset + 1 AND 4 * (f.year - ep.t) + 3 <= s.entry_offset + 4)))
           OR (NOT s.precise AND f.year = year(s.entry_date) + 1)
        """
    )
    con.execute(
        """
        CREATE TABLE lab AS
        SELECT s.episode_id, s.author_id, s.precise, s.entry_offset,
               CASE WHEN s.precise THEN s.entry_offset + 5 ELSE 4 * (year(s.entry_date) - ep.t + 2) END AS u,
               f.author_id IS NOT NULL AS label
        FROM s1e s JOIN ep USING (episode_id) LEFT JOIN follow f USING (episode_id, author_id)
        """
    )
    # Recruited entries: the entry work has a coauthor whose first topic work is earlier and in the response years.
    con.execute(
        """
        CREATE TABLE recruit AS
        SELECT s.episode_id, s.author_id,
               COALESCE(bool_or(f.first_date < s.entry_date AND year(f.first_date) BETWEEN ep.t AND ep.t + 2), FALSE) AS recruited,
               COALESCE(bool_or(year(f.first_date) < ep.t), FALSE) AS incumbent_coauthor
        FROM s1e s JOIN ep USING (episode_id)
        JOIN aw e ON e.author_id = s.author_id AND e.topic = ep.topic AND e.date = s.entry_date
        JOIN a co ON co.work_id = e.work_id AND co.author_id <> s.author_id
        LEFT JOIN ft f ON f.author_id = co.author_id AND f.topic = ep.topic
        GROUP BY s.episode_id, s.author_id
        """
    )
    con.execute(
        """
        CREATE TABLE ent AS
        SELECT s.episode_id, s.author_id, s.precise, s.entry_offset, s.entry_offset // 4 AS entry_ry,
               COALESCE(r.recruited, FALSE) AS recruited, COALESCE(r.incumbent_coauthor, FALSE) AS incumbent_coauthor
        FROM s1e s LEFT JOIN recruit r USING (episode_id, author_id)
        """
    )
    # Dyad states at each row's quarter.
    con.execute(
        """
        CREATE TABLE xstate AS
        SELECT r.episode_id, r.author_id, r.ry, r.quarter, r.event,
               COUNT(c.j) AS n_visible, MIN(c.v) AS first_v,
               COALESCE(bool_or(l.author_id IS NOT NULL AND l.u <= r.quarter AND l.label), FALSE) AS s,
               COALESCE(bool_or(l.author_id IS NOT NULL AND l.u <= r.quarter AND NOT l.label), FALSE) AS f,
               COALESCE(bool_or(l.author_id IS NOT NULL AND l.u > r.quarter AND l.label), FALSE) AS p_s,
               COALESCE(bool_or(l.author_id IS NOT NULL AND l.u > r.quarter AND NOT l.label), FALSE) AS p_f,
               COALESCE(bool_or(c.j IS NOT NULL AND l.author_id IS NULL), FALSE) AS u
        FROM xrows r
        LEFT JOIN dyad d ON d.episode_id = r.episode_id AND d.i = r.author_id AND d.ry = r.ry
        LEFT JOIN cent c ON c.episode_id = d.episode_id AND c.j = d.j AND c.v <= r.quarter
        LEFT JOIN lab l ON l.episode_id = c.episode_id AND l.author_id = c.j
        GROUP BY r.episode_id, r.author_id, r.ry, r.quarter, r.event
        """
    )
    parity["discovery_entrants_without_a_label"] = _scalar(con, "SELECT COUNT(*) FROM s1e ANTI JOIN lab USING (episode_id, author_id)")
    parity["state_rows_differing_from_structure"] = _scalar(con, """
        SELECT COUNT(*) FROM xstate s JOIN xvis v USING (episode_id, author_id, quarter)
        WHERE s.n_visible <> v.n_visible OR s.event <> v.event OR s.first_v IS DISTINCT FROM v.first_v""")

    model = estimator["state_model"]["controls"]
    features = estimator["features"]
    con.execute(
        """
        CREATE TABLE arows AS
        SELECT x.episode_id, x.author_id, x.quarter, x.event, 'one' AS weight_class,
               f.productivity_works, f.subfield_works, f.career_age_years, f.incumbent_coauthors,
               x.n_visible, CASE WHEN x.n_visible > 0 THEN x.quarter - x.first_v ELSE -1 END AS lag,
               x.s::INT AS s, x.f::INT AS f, x.p_s::INT AS p_s, x.p_f::INT AS p_f, x.u::INT AS u
        FROM xstate x JOIN xfeat f USING (episode_id, author_id, ry)
        UNION ALL
        SELECT np.episode_id, np.author_id, np.quarter, np.event, CASE WHEN np.entrant THEN 'one' ELSE 'pi' END,
               np.productivity_works, np.subfield_works, np.career_age_years, np.incumbent_coauthors,
               np.coauthor_entries_visible, -1, 0, 0, 0, 0, 0
        FROM np ANTI JOIN xa USING (episode_id, author_id)
        """
    )
    # Censoring: imprecise entrants from the start of the entry year; recruited precise entrants from the entry quarter.
    con.execute(
        """
        CREATE TABLE crow AS
        SELECT r.*, COALESCE(e.precise, TRUE) AS precise, COALESCE(e.recruited, FALSE) AS recruited,
               e.entry_offset, e.entry_ry
        FROM arows r LEFT JOIN ent e USING (episode_id, author_id)
        WHERE NOT (e.author_id IS NOT NULL AND NOT e.precise AND r.quarter // 4 >= e.entry_ry)
        """
    )
    prod_case = _case_sql("productivity_works", features["productivity_works"])
    age_case = _case_sql("career_age_years", features["career_age_years"])
    n_case = _case_sql("n_visible", model["visible_coauthor_entries"], below=1)
    lag_case = _case_sql("lag", model["quarters_since_the_first_visible_entry"], below=0)
    parity["rows_outside_a_registered_bin"] = _scalar(con, f"""
        SELECT COUNT(*) FROM crow WHERE ({prod_case}) IS NULL OR ({age_case}) IS NULL OR ({n_case}) IS NULL OR ({lag_case}) IS NULL""")
    cells = con.execute(
        f"""
        SELECT episode_id, quarter, {prod_case} AS prod_bin,
               CASE WHEN subfield_works = 0 THEN 0 WHEN 2 * subfield_works < productivity_works THEN 1 ELSE 2 END AS sub_bin,
               {age_case} AS age_bin, (incumbent_coauthors > 0)::INT AS incumbent,
               {n_case} AS n_bin, {lag_case} AS lag_bin, s, f, p_s, p_f, u, weight_class,
               COUNT(*) FILTER (WHERE NOT (recruited AND entry_offset IS NOT NULL AND quarter >= entry_offset)) AS rows,
               COUNT(*) FILTER (WHERE event = 1 AND NOT recruited) AS events,
               COUNT(*) AS rows_all, COUNT(*) FILTER (WHERE event = 1) AS events_all
        FROM crow
        GROUP BY ALL ORDER BY ALL
        """
    ).to_arrow_table().cast(CELL_SCHEMA)
    entry_cells = con.execute(
        """
        SELECT l.episode_id, l.entry_offset, COUNT(*) AS entrants, COUNT(*) FILTER (WHERE l.label) AS labelled
        FROM lab l WHERE l.precise GROUP BY ALL ORDER BY ALL
        """
    ).fetchall()
    counts = {
        "discovery_episodes": _scalar(con, "SELECT COUNT(*) FROM ep"),
        "discovery_entrants": _scalar(con, "SELECT COUNT(*) FROM s1e"),
        "discovery_precise_entrants": _scalar(con, "SELECT COUNT(*) FROM s1e WHERE precise"),
        "labelled_precise_entrants": _scalar(con, "SELECT COUNT(*) FROM lab WHERE precise AND label"),
        "labelled_imprecise_entrants": _scalar(con, "SELECT COUNT(*) FROM lab WHERE NOT precise AND label"),
        "recruited_precise_entrants": _scalar(con, "SELECT COUNT(*) FROM ent WHERE precise AND recruited"),
        "precise_entrants_with_an_incumbent_entry_coauthor": _scalar(con, "SELECT COUNT(*) FROM ent WHERE precise AND incumbent_coauthor"),
        "coauthor_entrants": _scalar(con, "SELECT COUNT(*) FROM cent"),
        "dyads": _scalar(con, "SELECT COUNT(*) FROM dyad"),
        "dyads_with_an_s1_coauthor": _scalar(con, "SELECT COUNT(*) FROM dyad d JOIN lab l ON l.episode_id = d.episode_id AND l.author_id = d.j"),
        "exposed_authors": _scalar(con, "SELECT COUNT(*) FROM xa"),
        "exposed_rows": _scalar(con, "SELECT COUNT(*) FROM xstate"),
        "exposed_rows_with_a_visible_entry": _scalar(con, "SELECT COUNT(*) FROM xstate WHERE n_visible > 0"),
        "n11b_discovery_rows_replaced": _scalar(con, "SELECT COUNT(*) FROM np JOIN xa USING (episode_id, author_id)"),
    }
    probability = {int(e): float(p) for e, p in con.execute("SELECT episode_id, sampling_probability FROM samp ORDER BY episode_id").fetchall()}
    con.close()
    return {"cells": cells, "entry_cells": entry_cells, "probability": probability, "parity": dict(sorted(parity.items())), "counts": counts}


def _scalar(con: Any, query: str) -> int:
    return int(con.execute(query).fetchone()[0] or 0)


def _case_sql(column: str, edges: Sequence[Sequence[int | None]], below: int | None = None) -> str:
    parts = []
    if below is not None:
        parts.append(f"WHEN {column} < {below} THEN -1")
    for index, (low, high) in enumerate(edges):
        condition = " AND ".join(c for c in (f"{column} >= {low}" if low is not None else "", f"{column} <= {high}" if high is not None else "") if c)
        parts.append(f"WHEN {condition or 'TRUE'} THEN {index}")
    return "CASE " + " ".join(parts) + " END"


def empirical_cells(built: Mapping[str, Any], estimator: Mapping[str, Any]) -> dict[str, Cells]:
    table = built["cells"].to_pydict()
    arrays = {key: np.asarray(value) for key, value in table.items()}
    probability = built["probability"]
    episode = arrays["episode_id"].astype(np.int64)
    weight = np.where(arrays["weight_class"] == "pi", np.asarray([1.0 / probability[int(e)] for e in episode]), 1.0)
    group = episode * 12 + arrays["quarter"].astype(np.int64)
    features = estimator["features"]
    prod, prod_names = _one_hot(arrays["prod_bin"].astype(np.int64), len(features["productivity_works"]), "prod")
    sub, sub_names = _one_hot(arrays["sub_bin"].astype(np.int64), 3, "sub")
    age, age_names = _one_hot(arrays["age_bin"].astype(np.int64), len(features["career_age_years"]), "age")
    base = np.column_stack([prod, sub, age, arrays["incumbent"].astype(np.float64)])
    base_names = [*prod_names, *sub_names, *age_names, "incumbent"]
    exposed = (arrays["n_bin"] >= 0).astype(np.float64)
    out = {}
    for key, rows, events in (("exposure", "rows", "events"), ("exposure_recruited_as_events", "rows_all", "events_all")):
        out[key] = collapse(group, episode, np.column_stack([exposed, base]), arrays[rows] * weight, arrays[events] * weight, ["exposed", *base_names])
    keep = arrays["n_bin"] >= 0
    require(bool(np.all(weight[keep] == 1.0)), "every row with a visible coauthor entry belongs to an exposed author")
    states = {state: arrays[state.lower()][keep].astype(np.int64) for state in STATES}
    design, names, reference = _state_design(states, arrays["n_bin"][keep].astype(np.int64), arrays["lag_bin"][keep].astype(np.int64), estimator)
    events = arrays["events"][keep].astype(np.float64)
    state_events = {s: float(events[states[s] == 1].sum()) for s in STATES}
    state_events["reference"] = reference
    out["state"] = collapse(group[keep], episode[keep], np.column_stack([design, base[keep]]), arrays["rows"][keep].astype(np.float64), events,
                            [*names, *base_names], state_events)
    entry = np.asarray(built["entry_cells"], dtype=np.float64).reshape(-1, 4)
    out["entry_order"] = collapse(entry[:, 0].astype(np.int64), entry[:, 0].astype(np.int64), (entry[:, 1] / 11.0)[:, None], entry[:, 2],
                                  entry[:, 3], ["entry_time"])
    return out


def empirical_problem(contract: Mapping[str, Any], cells: Mapping[str, Cells]) -> tuple[dict[str, Any], list]:
    estimator = contract["estimator"]
    settings = _settings(estimator)
    models = []
    for model_key, quantities in (("exposure", ["beta:exposed"]), ("exposure_recruited_as_events", ["beta:exposed"]),
                                  ("state", list(CONTRASTS)), ("entry_order", ["beta:entry_time"])):
        model_cells = cells[model_key]
        fit = fit_logit(model_cells, **settings)
        require(fit["converged"], f"a full-sample data fit did not converge: {model_key}")
        models.append({"arm": "data", "model": model_key, "cells": model_cells, "beta": fit["beta"],
                       "reference": model_cells.state_events.get("reference", "P_F"), "quantities": quantities})
    episodes = sorted({int(c) for model in models for c in model["cells"].cluster})
    index = {e: k for k, e in enumerate(episodes)}
    for model in models:
        model["cells"] = Cells(group=model["cells"].group, cluster=np.asarray([index[int(c)] for c in model["cells"].cluster]),
                               trials=model["cells"].trials, events=model["cells"].events, design=model["cells"].design,
                               names=model["cells"].names, state_events=model["cells"].state_events)
    draws = int(estimator["bootstrap"]["draws"])
    problem = {"models": models, "multiplicities": multiplicities(len(episodes), draws, int(estimator["bootstrap"]["seed"]), 2)}
    return {"data": problem}, [_point_values(m) for m in models]


def data_report(problem: Mapping[str, Any], points: Sequence, draws: Mapping[str, Any]) -> dict[str, Any]:
    values = draws["draws"]
    column = 0
    out: dict[str, Any] = {}
    for model, point in zip(problem["models"], points, strict=True):
        for quantity, value in zip(model["quantities"], point, strict=True):
            name = quantity[5:] if quantity.startswith("beta:") else quantity
            out[f"{model['model']}/{name}"] = summarize_draws(value if _quantity_defined(model, quantity) else None, values[:, column])
            column += 1
    state = next(m for m in problem["models"] if m["model"] == "state")
    return {"estimates": out, "state_events": dict(state["cells"].state_events), "unconverged_fits": draws["unconverged_fits"]}


# -- decisions ---------------------------------------------------------------------------


def _low_above(row: Mapping[str, Any] | None) -> bool:
    return bool(row) and row.get("low") is not None and row["low"] > 0


def _high_below(row: Mapping[str, Any] | None) -> bool:
    return bool(row) and row.get("high") is not None and row["high"] < 0


def _covers_zero(row: Mapping[str, Any] | None) -> bool | None:
    if not row or row.get("low") is None:
        return None
    return row["low"] <= 0 <= row["high"]


def decide(contract: Mapping[str, Any], simulation: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = contract["simulation"]["scenarios"]
    es2_rows = {s: simulation[f"simulation/{s}/delay_0"]["learning_attributable"].get("imitative", {}).get("state/outcome_gap_visible")
                for s in ("baseline", "interdisciplinary_seed_grants")}
    es3_rows = {s: simulation[f"simulation/{s}/delay_3"]["by_arm"]["frozen"]["state/visibility_did"] for s in scenarios}
    covered = {s: _covers_zero(row) for s, row in es3_rows.items()}
    defined = [s for s, value in covered.items() if value is not None]
    es3 = len(defined) >= 2 and all(covered[s] for s in defined)
    estimates = data["estimates"]
    expectations = {
        "ES2": {"held": all(_low_above(row) for row in es2_rows.values()), "rows": es2_rows},
        "ES3": {"held": es3, "covers_zero": covered, "defined_in": defined, "rows": es3_rows},
        "EE1": {"held": _low_above(estimates["exposure/exposed"]), "row": estimates["exposure/exposed"]},
        "EE2": {"held": _low_above(estimates["state/outcome_gap_before_visible"]), "row": estimates["state/outcome_gap_before_visible"]},
    }
    did = estimates["state/visibility_did"]
    following, avoidance = estimates["state/success_following"], estimates["state/failure_avoidance"]
    if not es3:
        disposition = "NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM"
    elif did["mean"] is None:
        disposition = "UNDEFINED_TOO_FEW_EVENTS"
    elif _low_above(did):
        if _low_above(following) and _high_below(avoidance):
            disposition = "CONSISTENT_WITH_IMITATIVE"
        elif _high_below(avoidance):
            disposition = "CONSISTENT_WITH_CAUTIONARY"
        elif _low_above(following):
            disposition = "SUCCESS_FOLLOWING_ONLY_NO_STUDY2_ARM"
        else:
            disposition = "OUTCOME_DEPENDENCE_EMERGES_WITHOUT_A_DIRECTION"
    elif _high_below(did):
        disposition = "OUTCOME_ASSOCIATION_WEAKENS_ONCE_VISIBLE"
    else:
        disposition = "NO_VISIBILITY_EFFECT_DETECTED"
    return {"status": f"E4_{disposition}", "disposition": disposition, "expectations": expectations}


# -- run ---------------------------------------------------------------------------------


def n11b_inputs(contract: Mapping[str, Any], n11b: ModuleType) -> dict[str, Any]:
    n11b_contract_path = _binding(contract, N11B_CONTRACT_ROLE)
    n11b_contract = n11b.load_contract(n11b_contract_path)
    s1 = n11b.load_s1(n11b_contract)
    preflight = n11b.preflight(n11b_contract, s1)
    guard = n11b.s1_contract(n11b_contract, s1)["build"]["g4_guard"]
    root = Path(n11b_contract["inputs"]["s1_root"])
    n11b_root = Path(contract["inputs"]["n11b_root"])
    receipt_ok = _sha256(n11b_root / "n11b_receipt.json") == contract["inputs"]["n11b_receipt_sha256"]
    receipt = json.loads((n11b_root / "n11b_receipt.json").read_text(encoding="utf-8"))
    drift = [name for name in contract["inputs"]["n11b_files"] if _sha256(n11b_root / name) != receipt["outputs"][name]]
    return {
        "preflight": preflight,
        "n11b_receipt_identity": receipt_ok,
        "n11b_output_drift": drift,
        "inputs": {name: root / relative for name, relative in n11b.INPUTS.items()},
        "guard_path": Path(guard["path"]),
        "guard_column": guard["column"],
        "n11b_private": {"panel": n11b_root / "private" / "panel.parquet", "episodes_sampling": n11b_root / "private" / "episodes_sampling.parquet"},
        "passes": preflight["status"].startswith("PASS") and receipt_ok and not drift,
    }


def run(contract: Mapping[str, Any], *, workers: int, threads: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    timing: dict[str, float] = {}
    started = time.monotonic()
    study2 = load_study2(contract)
    tracked = json.loads(_binding(contract, TRACKED_ROLE).read_text(encoding="utf-8"))
    simulation = run_simulation(contract, study2, tracked)
    timing["simulation_runs"] = round(time.monotonic() - started, 1)
    require(simulation["parity"]["passes"], f"Study 2 parity failed: {simulation['parity']}")

    n11b = load_n11b(contract)
    checks = n11b_inputs(contract, n11b)
    require(checks["passes"], f"the N11b inputs failed their checks: {checks['preflight']['status']} {checks['n11b_output_drift']}")
    mark = time.monotonic()
    built = build_empirical(checks["inputs"], checks["guard_path"], checks["guard_column"], checks["n11b_private"], n11b,
                            contract["estimator"], threads=threads)
    timing["empirical_build"] = round(time.monotonic() - mark, 1)
    require(not any(built["parity"].values()), f"N11b parity failed: {built['parity']}")

    mark = time.monotonic()
    sim_problems, sim_points = simulation_problems(contract, simulation)
    data_cells = empirical_cells(built, contract["estimator"])
    data_problems, data_points = empirical_problem(contract, data_cells)
    draws = bootstrap({**sim_problems, **data_problems}, _settings(contract["estimator"]), workers=workers)
    timing["estimation"] = round(time.monotonic() - mark, 1)
    sim_report = simulation_report(sim_problems, sim_points, draws)
    data = data_report(data_problems["data"], data_points, draws["data"])
    decision = decide(contract, sim_report, data)
    summary = {
        "schema_version": "pivot_peer_outcome_contrast_summary.v1",
        "status": decision["status"],
        "decision": decision,
        "checks": {
            "study2_parity": simulation["parity"],
            "n11b_preflight": checks["preflight"]["status"],
            "n11b_receipt_identity": checks["n11b_receipt_identity"],
            "n11b_output_drift": checks["n11b_output_drift"],
            "n11b_parity": built["parity"],
        },
        "data": {**data, "counts": built["counts"]},
        "simulation": sim_report,
        "boundary": dict(contract["boundary"]),
        "data_access": {"partition": "DISCOVERY_ONLY", "confirmatory_outcome_rows_read": 0, "g4_outcome_rows_read": 0},
    }
    timing["total"] = round(time.monotonic() - started, 1)
    return summary, built, timing


def write_outputs(output_dir: Path, summary: Mapping[str, Any], built: Mapping[str, Any], *, contract_path: Path, authorization_path: Path,
                  precommit: Mapping[str, Any], timing: Mapping[str, float]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    output_dir.chmod(0o700)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    pq.write_table(built["cells"], private / "cells.parquet")
    (private / "cells.parquet").chmod(0o600)
    _write_exclusive(output_dir / "e4_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "pivot_peer_outcome_contrast_receipt.v1",
        "status": "COMPLETE",
        "contract_sha256": _sha256(contract_path),
        "script_sha256": _sha256(Path(__file__)),
        "authorization_sha256": _sha256(authorization_path),
        "precommit": dict(precommit),
        "outputs": {name: _sha256(output_dir / name) for name in ("e4_summary.json", "private/cells.parquet")},
    }
    _write_exclusive(output_dir / "e4_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write_exclusive(output_dir / "e4_timing.json", json.dumps(dict(timing), indent=2, sort_keys=True) + "\n")
    return receipt


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    load_authorization(args.authorization, config_path, Path(__file__).resolve())
    if args.command == "plan":
        n11b = load_n11b(contract)
        checks = n11b_inputs(contract, n11b)
        print(json.dumps({"status": "PRECOMMIT_PLAN_VALID", "contract_sha256": _sha256(config_path), "n11b_preflight": checks["preflight"]["status"],
                          "n11b_receipt_identity": checks["n11b_receipt_identity"], "n11b_output_drift": checks["n11b_output_drift"]},
                         indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or args.output_dir is None:
        parser.error("run requires --precommit-sha and --output-dir")
    precommit = verify_precommit(args.precommit_sha, contract)
    summary, built, timing = run(contract, workers=args.workers, threads=args.threads)
    summary = {**summary, "precommit_sha": args.precommit_sha}
    receipt = write_outputs(args.output_dir.resolve(), summary, built, contract_path=config_path, authorization_path=args.authorization.resolve(),
                            precommit=precommit, timing=timing)
    print(json.dumps({"status": summary["status"], "receipt": receipt["status"], "wall_seconds": timing["total"]}))


if __name__ == "__main__":
    main()
