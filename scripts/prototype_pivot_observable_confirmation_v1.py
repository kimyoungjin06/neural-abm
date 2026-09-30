#!/usr/bin/env python3
"""Run the precommitted Pivot observable-confirmation study v1 (N08).

A third, registered test of the two observables that separated every N05
cell, with thresholds fixed from N05, on new hazard-style and surge-style
entry schedules and fresh seeds. A no-learning control checks that the
observables read learning, not entry timing alone.
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

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_observable_confirmation_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_observable_discriminant_v1 as n05  # noqa: E402


rdc = n05.rdc
abm1 = n05.abm1
abm2 = n05.abm2
delayed = n05.delayed
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = n05.FIT_WORLDS
VISIBILITY_IDS = n05.VISIBILITY_IDS
MAIN_ARMS = n05.MAIN_ARMS
FEATURE_ARMS = n05.FEATURE_ARMS
CONFIRMED = ("cohort_persistence_drop", "outcome_selection_trend")
OPENED_SEEDS = n05.OPENED_SEEDS | {20261031, 20261032}


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("observable-confirmation contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_observable_confirmation.v1":
        raise ValueError("unexpected observable-confirmation schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("observable-confirmation contract is not a precommit design")
    if [row["id"] for row in contract["visibility_regimes"]] != list(VISIBILITY_IDS):
        raise ValueError("visibility regimes changed")
    if [tuple(row) for row in contract["main_run_arms"]] != list(MAIN_ARMS):
        raise ValueError("main-run arms changed")
    thresholds = contract["thresholds"]
    n05_ranges = thresholds["n05_separation"]
    drop = n05_ranges["cohort_persistence_drop"]
    trend = n05_ranges["outcome_selection_trend"]
    if float(thresholds["cohort_persistence_drop"]["threshold"]) != round((drop["active_max"] + drop["inactive_min"]) / 2, 6):
        raise ValueError("the persistence-drop threshold must be the N05 midpoint")
    if float(thresholds["outcome_selection_trend"]["threshold"]) != round((trend["active_min"] + trend["inactive_max"]) / 2, 6):
        raise ValueError("the outcome-trend threshold must be the N05 midpoint")
    opened = {(row["type"], tuple(sorted((k, v) for k, v in row.items() if k not in ("id", "type")))) for row in contract["opened_schedules"]}
    for row in contract["entry_schedules"]:
        key = (row["type"], tuple(sorted((k, v) for k, v in row.items() if k not in ("id", "type"))))
        if key in opened:
            raise ValueError(f"schedule {row['id']} repeats an opened N01 or N05 schedule")
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
    boundaries = contract["boundaries"]
    if boundaries.get("empirical_rows_read") != 0 or boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("the study reads no empirical rows and changes no generic API")


def schedules(contract: Mapping[str, Any], profile: str) -> dict[str, tuple[float, ...]]:
    config = abm1.study_config(contract, profile)
    open_steps = config.steps - config.burn_in_steps
    floor = float(contract["surge_floor"])
    built = {}
    for row in contract["entry_schedules"]:
        if row["type"] == "hazard":
            built[row["id"]] = (float(row["q"]),) * open_steps
        elif row["type"] == "surge":
            built[row["id"]] = n05.surge_schedule(float(row["peak"]), float(row["width"]), floor, open_steps)
        else:
            raise ValueError(f"unknown schedule type: {row['type']}")
    return built


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


# -- decisions ---------------------------------------------------------------------------


def _mean(values: Sequence[Any]) -> float | None:
    kept = [float(value) for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    return float(np.mean(kept)) if kept else None


def predict(name: str, value: float | None, thresholds: Mapping[str, Any]) -> bool:
    threshold = float(thresholds[name]["threshold"])
    if value is None:
        return False
    return value < threshold if thresholds[name]["active_when"] == "BELOW" else value > threshold


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {(int(r["replicate"]), r["clock_id"], r["visibility_id"], r["world_id"], r["arm_id"]): r for r in rows}
    replicates = sorted({int(r["replicate"]) for r in rows})
    thresholds = contract["thresholds"]
    tiers = thresholds["accuracy_tiers"]
    delays = {row["id"]: int(row["failure_delay_steps"]) for row in contract["visibility_regimes"]}
    cells = []
    for schedule in contract["entry_schedules"]:
        for visibility in VISIBILITY_IDS:
            selected = [r for r in rows if r["clock_id"] == schedule["id"] and r["visibility_id"] == visibility]
            feature = [r for r in selected if r["arm_id"] in FEATURE_ARMS]
            fixed = [r for r in selected if r["arm_id"] == "fixed_rule"]
            cells.append(
                {
                    "schedule_id": schedule["id"], "schedule_type": schedule["type"],
                    "visibility_id": visibility, "delay": delays[visibility],
                    **n05.cell_status(indexed, replicates, contract, schedule["id"], visibility),
                    "post_failure_entry_share": _mean([r["post_failure_entry_share"] for r in feature]),
                    "interior_peak_share": _mean([float(r["interior_entry_peak"]) for r in feature]),
                    **{name: _mean([r[name] for r in feature]) for name in CONFIRMED},
                    **{f"fixed_rule_{name}": _mean([r[name] for r in fixed]) for name in CONFIRMED},
                }
            )
    active = [cell["status"] == "ENTRY_CHANNEL_ACTIVE" for cell in cells]
    tests = {}
    for name in CONFIRMED:
        predicted = [predict(name, cell[name], thresholds) for cell in cells]
        tests[name] = {
            "threshold": float(thresholds[name]["threshold"]),
            "active_when": thresholds[name]["active_when"],
            **n05._accuracy(predicted, active, tiers),
            "separation_in_this_study": n05.separation([cell[name] for cell in cells], active),
        }
    both = [all(predict(name, cell[name], thresholds) for name in CONFIRMED) for cell in cells]
    control = {
        name: {
            "cells_reading_inactive": sum(not predict(name, cell[f"fixed_rule_{name}"], thresholds) for cell in cells),
            "cells": len(cells),
        }
        for name in CONFIRMED
    }
    control_ok = all(entry["cells_reading_inactive"] == entry["cells"] for entry in control.values())
    pfes_threshold = float(thresholds["post_failure_entry_share_comparator"]["threshold"])
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "cells": cells,
        "status_mix": {"active_cells": sum(active), "inactive_cells": len(active) - sum(active), "discriminating": 0 < sum(active) < len(active)},
        "observable_tests": tests,
        "joint_rule_both_observables": n05._accuracy(both, active, tiers),
        "no_learning_control": {
            **control,
            "disposition": "CONTROL_READS_INACTIVE_EVERYWHERE" if control_ok else "CONTROL_READS_ACTIVE_SOMEWHERE_OBSERVABLES_MAY_TRACK_TIMING",
        },
        "post_failure_entry_share_comparator": {
            "threshold": pfes_threshold,
            **n05._accuracy([c["post_failure_entry_share"] is not None and c["post_failure_entry_share"] >= pfes_threshold for c in cells], active, tiers),
        },
        "mean_interior_peak_share": _mean([cell["interior_peak_share"] for cell in cells]),
    }


def summarize(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    config = abm1.study_config(contract, profile)
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    return {
        "schema_version": "pivot_observable_confirmation_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "schedules": {key: [round(v, 6) for v in value] for key, value in schedules(contract, profile).items()},
        "abm_runs": len(rows),
        "dispositions": {"status": "NOT_EVALUATED_SMOKE_PROFILE"} if config.replicates < required else decide(rows, contract),
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Pivot observable-confirmation scorecard v1", "", f"Profile: `{summary['profile']}`", ""]
    d = summary["dispositions"]
    if "cells" in d:
        lines.extend(
            [
                "| Schedule | Visibility | Status | Persistence drop | Outcome trend | Fixed-rule drop | Fixed-rule trend |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        fmt = lambda x: "-" if x is None else f"{x:+.3f}"  # noqa: E731
        for cell in d["cells"]:
            lines.append(
                f"| {cell['schedule_id']} | {cell['visibility_id']} | {cell['status']} | {fmt(cell['cohort_persistence_drop'])} | "
                f"{fmt(cell['outcome_selection_trend'])} | {fmt(cell['fixed_rule_cohort_persistence_drop'])} | "
                f"{fmt(cell['fixed_rule_outcome_selection_trend'])} |"
            )
        lines.extend(["", "## Frozen dispositions", ""])
        for name, entry in d["observable_tests"].items():
            lines.append(f"- {name}: `{entry['disposition']}` ({entry['correct']}/{entry['cells']})")
        lines.append(f"- both observables: `{d['joint_rule_both_observables']['disposition']}`")
        lines.append(f"- no-learning control: `{d['no_learning_control']['disposition']}`")
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


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
        "schema_version": "pivot_observable_confirmation_receipt.v1",
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
