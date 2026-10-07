#!/usr/bin/env python3
"""Run the precommitted Pivot one-sided observable test v1 (N09).

N08 found that the two persistence-trend observables never called an
inactive cell active. This study registers that one-sided claim, with the
N08 thresholds unchanged, on new entry-schedule shapes and fresh seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
import yaml
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_observable_one_sided_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_observable_confirmation_v1 as n08  # noqa: E402


n05 = n08.n05
rdc = n08.rdc
abm1 = n08.abm1
abm2 = n08.abm2
delayed = n08.delayed
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = n08.FIT_WORLDS
VISIBILITY_IDS = n08.VISIBILITY_IDS
MAIN_ARMS = n08.MAIN_ARMS
FEATURE_ARMS = n08.FEATURE_ARMS
OBSERVABLES = n08.CONFIRMED
SCHEDULE_TYPES = ("hazard", "surge", "decay", "ramp")
OPENED_SEEDS = n08.OPENED_SEEDS | {20261033, 20261034}
N08_CONTRACT = ROOT / "experiments" / "configs" / "pivot_observable_confirmation_v1.yaml"


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("one-sided contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_observable_one_sided.v1":
        raise ValueError("unexpected one-sided schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("one-sided contract is not a precommit design")
    if [row["id"] for row in contract["visibility_regimes"]] != list(VISIBILITY_IDS):
        raise ValueError("visibility regimes changed")
    if [tuple(row) for row in contract["main_run_arms"]] != list(MAIN_ARMS):
        raise ValueError("main-run arms changed")
    n08_contract = yaml.safe_load(N08_CONTRACT.read_text(encoding="utf-8"))
    for name in OBSERVABLES:
        if contract["thresholds"][name] != n08_contract["thresholds"][name]:
            raise ValueError(f"the {name} threshold must be the N08 registered one, unchanged")
    if any(row["type"] not in SCHEDULE_TYPES for row in contract["entry_schedules"]):
        raise ValueError("unknown schedule type")
    opened = {_schedule_key(row) for row in contract["opened_schedules"]}
    for row in contract["entry_schedules"]:
        if _schedule_key(row) in opened:
            raise ValueError(f"schedule {row['id']} repeats an opened schedule")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from v2")
    seeds = {int(contract["profiles"]["development"]["base_seed"]), int(contract["probe_cohort"]["base_seed"])}
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("the study needs fresh seed namespaces")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != rdc.IMPROVEMENT_RULE or decisions.get("difference_in_differences_rule") != rdc.DID_RULE:
        raise ValueError("the N01 entry-channel rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this study")
    if contract["boundaries"].get("empirical_rows_read") != 0:
        raise ValueError("the study reads no empirical rows")


def _schedule_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["type"], tuple(sorted((k, float(v)) for k, v in row.items() if k not in ("id", "type"))))


def build_schedule(row: Mapping[str, Any], floor: float, open_steps: int) -> tuple[float, ...]:
    kind = row["type"]
    if kind == "hazard":
        return (float(row["q"]),) * open_steps
    if kind == "surge":
        return n05.surge_schedule(float(row["peak"]), float(row["width"]), floor, open_steps)
    if kind == "decay":
        tau = float(row["tau"])
        return tuple(float(floor + (1.0 - floor) * math.exp(-(step - 1) / tau)) for step in range(1, open_steps + 1))
    if kind == "ramp":
        length = float(row["ramp_steps"])
        return tuple(float(floor + (1.0 - floor) * min(1.0, step / length)) for step in range(1, open_steps + 1))
    raise ValueError(f"unknown schedule type: {kind}")


def schedules(contract: Mapping[str, Any], profile: str) -> dict[str, tuple[float, ...]]:
    config = abm1.study_config(contract, profile)
    open_steps = config.steps - config.burn_in_steps
    floor = float(contract["schedule_floor"])
    return {row["id"]: build_schedule(row, floor, open_steps) for row in contract["entry_schedules"]}


def run_task(
    contract: Mapping[str, Any], profile: str, replicate: int, schedule_id: str, visibility_id: str, world_id: str
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    schedule = schedules(contract, profile)[schedule_id]
    horizon = int(contract["h3_adapter"]["cohort_persistence_horizon_steps"])
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows = []
    for arm_id, evidence_regime in MAIN_ARMS:
        adapter_class = n05.surge_adapter(schedule)
        with n05.adapter_slot(adapter_class):
            run = rdc.run_arm(
                contract, config, replicate=replicate, visibility_id=visibility_id, world_id=world_id,
                arm_id=arm_id, evidence_regime=evidence_regime, probe=probe, q=1.0,
            )
        if rdc.HazardCommitAdapter is not n05.ORIGINAL_ADAPTER or len(adapter_class.created) != 1:
            raise RuntimeError("the schedule adapter slot was not used once and restored")
        rows.append(
            {
                "replicate": replicate, "clock_id": schedule_id, "visibility_id": visibility_id,
                "world_id": world_id, "arm_id": arm_id, "evidence_regime": evidence_regime,
                **run,
                **n05.run_observables(run, adapter_class.created[0], horizon=horizon, steps=config.steps),
            }
        )
    return rows


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_study(
    contract: Mapping[str, Any], profile: str, workers: int = 1,
    schedule_ids: Sequence[str] | None = None, visibilities: Sequence[str] = VISIBILITY_IDS,
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    ids = list(schedule_ids) if schedule_ids is not None else [row["id"] for row in contract["entry_schedules"]]
    tasks = [
        (contract, profile, replicate, schedule_id, visibility_id, world_id)
        for replicate in range(config.replicates)
        for schedule_id in ids
        for visibility_id in visibilities
        for world_id in FIT_WORLDS
    ]
    if workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            results = [_run_task_args(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init) as pool:
            results = list(pool.map(_run_task_args, tasks))
    return [row for rows in results for row in rows]


def one_sided(predicted: Sequence[bool], active: Sequence[bool], rule: Mapping[str, Any]) -> dict[str, Any]:
    flagged = sum(predicted)
    false_active = sum(p and not a for p, a in zip(predicted, active, strict=True))
    detected = sum(p and a for p, a in zip(predicted, active, strict=True))
    informative = flagged >= int(rule["minimum_flagged_cells"]) and (len(active) - sum(active)) >= int(rule["minimum_inactive_cells"])
    if not informative:
        disposition = "NOT_INFORMATIVE"
    elif false_active == 0:
        disposition = "SUFFICIENT_SIGN_CONFIRMED"
    else:
        disposition = "SUFFICIENT_SIGN_FAILS"
    return {
        "flagged_cells": flagged,
        "false_active": false_active,
        "detected_active": detected,
        "active_cells": sum(active),
        "sensitivity": detected / sum(active) if sum(active) else None,
        "disposition": disposition,
    }


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    shim = {**contract, "entry_schedules": [{"id": row["id"], "type": row["type"]} for row in contract["entry_schedules"]]}
    base = n08.decide(rows, shim)
    thresholds = contract["thresholds"]
    rule = contract["decision_contract"]["one_sided"]
    active = [cell["status"] == "ENTRY_CHANNEL_ACTIVE" for cell in base["cells"]]
    sided = {
        name: one_sided([n08.predict(name, cell[name], thresholds) for cell in base["cells"]], active, rule)
        for name in OBSERVABLES
    }
    either = [any(n08.predict(name, cell[name], thresholds) for name in OBSERVABLES) for cell in base["cells"]]
    return {
        **base,
        "one_sided_tests": sided,
        "one_sided_either_observable": one_sided(either, active, rule),
    }


def summarize(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    config = abm1.study_config(contract, profile)
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    return {
        "schema_version": "pivot_observable_one_sided_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "schedules": {key: [round(v, 6) for v in value] for key, value in schedules(contract, profile).items()},
        "abm_runs": len(rows),
        "dispositions": {"status": "NOT_EVALUATED_SMOKE_PROFILE"} if config.replicates < required else decide(rows, contract),
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    text = n08._markdown(summary).replace("observable-confirmation", "one-sided observable")
    d = summary["dispositions"]
    if "one_sided_tests" in d:
        extra = ["", "## One-sided tests", ""]
        for name, entry in d["one_sided_tests"].items():
            extra.append(f"- {name}: `{entry['disposition']}` ({entry['false_active']} false actives of {entry['flagged_cells']} flagged)")
        either = d["one_sided_either_observable"]
        extra.append(f"- either observable: `{either['disposition']}` ({either['false_active']} false actives of {either['flagged_cells']} flagged)")
        text = text.replace("\nSynthetic worlds only.", "\n".join(extra) + "\n\nSynthetic worlds only.")
    return text


def write_outputs(
    *, output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], precommit: Mapping[str, Any], contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in files.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_observable_one_sided_receipt.v1",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_sha256": abm1._sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {name: abm1._sha256(output_dir / name) for name in files},
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": abm1._sha256(config_path),
        "schedules": {key: [round(v, 4) for v in value[:12]] for key, value in schedules(contract, "development").items()},
        "cells": len(contract["entry_schedules"]) * len(VISIBILITY_IDS),
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
    config = abm1.study_config(contract, args.profile)
    for replicate in range(config.replicates):
        n05.check_nesting(contract, args.profile, replicate)
    rows = run_study(contract, args.profile, workers=args.workers)
    summary = {**summarize(rows, contract, args.profile), "precommit_sha": args.precommit_sha}
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(), rows=rows, summary=summary, precommit=precommit, contract_path=config_path
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
