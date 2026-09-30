#!/usr/bin/env python3
"""Run the precommitted Pivot observable-discriminant study v1 (N05).

An attention-surge clock admits ready agents with a probability that rises and
falls over time, so entry peaks inside the window. The study tests the
model-internal post-failure entry share out of sample and screens
data-observable candidates for the same entry-channel status.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_observable_discriminant_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_relative_delay_cost_v1 as rdc  # noqa: E402


abm1 = rdc.abm1
abm2 = rdc.abm2
delayed = rdc.delayed
persistence = rdc.persistence
study1 = rdc.study1
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = rdc.FIT_WORLDS
INTERACTION_FIT_WORLDS = rdc.INTERACTION_FIT_WORLDS
VISIBILITY_IDS = rdc.VISIBILITY_IDS
MAIN_ARMS = rdc.MAIN_ARMS
FEATURE_ARMS = rdc.FEATURE_ARMS
NONLINEAR_ARMS = rdc.NONLINEAR_ARMS
ORIGINAL_ADAPTER = rdc.HazardCommitAdapter
SURGE_COMPONENT = 72
SATURATION_ENTRY = rdc.SATURATION_ENTRY
OPENED_SEEDS = rdc.OPENED_SEEDS | {
    20261014, 20261015, 20261016, 20261017, 20261018, 20261019, 20261020, 20261021,
    20261022, 20261023, 20261024, 20261025, 20261026, 20261027, 20261028, 20261029,
    20261030,
}
OBSERVABLES = (
    "post_decline_entry_share",
    "post_first_cohort_entry_share",
    "cohort_persistence_drop",
    "post_peak_entry_share",
    "attribute_selection_trend",
    "outcome_selection_trend",
)


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("observable-discriminant contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_observable_discriminant.v1":
        raise ValueError("unexpected observable-discriminant schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("observable-discriminant contract is not a precommit design")
    if [row["id"] for row in contract["visibility_regimes"]] != list(VISIBILITY_IDS):
        raise ValueError("visibility regimes changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if [tuple(row) for row in contract["main_run_arms"]] != list(MAIN_ARMS):
        raise ValueError("main-run arms changed")
    if tuple(contract["observables"]["registered"]) != OBSERVABLES:
        raise ValueError("the observable candidates changed")
    thresholds = contract["discriminants"]
    n01 = thresholds["n01_post_hoc_separation"]
    if float(thresholds["post_failure_entry_share"]["threshold"]) != round(
        (n01["post_failure_entry_share"]["min_active"] + n01["post_failure_entry_share"]["max_inactive"]) / 2, 6
    ):
        raise ValueError("the post-failure threshold must be the N01 midpoint")
    if float(thresholds["feature_arm_relative_delay"]["threshold"]) != round(
        (n01["feature_arm_relative_delay"]["max_active"] + n01["feature_arm_relative_delay"]["min_inactive"]) / 2, 6
    ):
        raise ValueError("the relative-delay threshold must be the N01 midpoint")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from v2")
    seeds = {int(contract["profiles"]["development"]["base_seed"]), int(contract["probe_cohort"]["base_seed"])}
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("the study needs fresh seed namespaces")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != rdc.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("difference_in_differences_rule") != rdc.DID_RULE:
        raise ValueError("difference-in-differences rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this study")
    boundaries = contract["boundaries"]
    if boundaries.get("empirical_rows_read") != 0 or boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("the study reads no empirical rows and changes no generic API")


# -- the attention-surge clock ---------------------------------------------------------


def surge_schedule(peak: float, width: float, floor: float, open_steps: int) -> tuple[float, ...]:
    """Admission probability by open step: a Gaussian surge above a floor."""

    return tuple(
        float(floor + (1.0 - floor) * math.exp(-0.5 * ((step - peak) / width) ** 2))
        for step in range(1, open_steps + 1)
    )


def clock_schedules(contract: Mapping[str, Any], profile: str) -> dict[str, tuple[float, ...]]:
    config = abm1.study_config(contract, profile)
    open_steps = config.steps - config.burn_in_steps
    spec = contract["surge_clock"]
    return {
        row["id"]: surge_schedule(float(row["peak"]), float(row["width"]), float(spec["floor"]), open_steps)
        for row in spec["clocks"]
    }


class SurgeCommitAdapter(ORIGINAL_ADAPTER):
    """The N01 hazard adapter with a time-varying admission probability."""

    schedule: tuple[float, ...] = ()
    created: list[Any] = []

    def __post_init__(self) -> None:
        super().__post_init__()
        type(self).created.append(self)

    def admission(self) -> float:
        open_step = self.environment.step - self.environment.config.burn_in_steps
        if 1 <= open_step <= len(self.schedule):
            return self.schedule[open_step - 1]
        return 1.0

    def admits(self, agent_id: int) -> bool:
        probability = self.admission()
        if probability >= 1.0:
            return True
        seed = np.random.SeedSequence(
            (self.base_seed, self.replicate, SURGE_COMPONENT, self.environment.step, agent_id)
        )
        return float(np.random.default_rng(seed).random()) < probability


def surge_adapter(schedule: Sequence[float]) -> type:
    return type(
        "TaskSurgeAdapter", (SurgeCommitAdapter,), {"schedule": tuple(schedule), "created": []}
    )


@contextlib.contextmanager
def adapter_slot(adapter_class: type) -> Iterator[None]:
    rdc.HazardCommitAdapter = adapter_class
    try:
        yield
    finally:
        rdc.HazardCommitAdapter = ORIGINAL_ADAPTER


# -- observables -----------------------------------------------------------------------


def _share_from(incidence: Sequence[int], step: int | None) -> float:
    total = sum(incidence)
    if step is None or not total or step > len(incidence):
        return 0.0
    return sum(incidence[max(step, 1) - 1 :]) / total


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return None if math.isnan(value) else value


def run_observables(run: Mapping[str, Any], adapter: Any, *, horizon: int, steps: int) -> dict[str, Any]:
    """The model-internal discriminant and every data-observable candidate for one run."""

    incidence = list(run["entry_incidence"])
    first_failure = next(
        (index + 1 for index, value in enumerate(run["public_failure_per_step"]) if value > 0), None
    )
    first_entry = next((index + 1 for index, value in enumerate(incidence) if value > 0), None)
    onset = run["decline_onset_step"]
    cohorts = run["cohort_persistence"]
    drop = None
    if len(cohorts) >= 2:
        shares = [persisting / count for _step, count, persisting in cohorts]
        drop = float(shares[0] - np.average(shares[1:], weights=[count for _s, count, _p in cohorts[1:]]))
    peak = min(run["entry_peak_steps"]) if run["entry_peak_steps"] else None
    entry_steps, fits, persisted = [], [], []
    for agent, entry, exit_step in zip(adapter.agents, adapter.entry_steps, adapter.exit_steps, strict=True):
        if entry is None:
            continue
        entry_steps.append(int(entry))
        fits.append(float(study1.Researcher.productive_fit.fget(agent.base)))
        if entry + horizon <= steps:
            persisted.append((int(entry), float(exit_step is None or exit_step > entry + horizon)))
    return {
        "first_public_failure_step": first_failure,
        "post_failure_entry_share": _share_from(incidence, first_failure),
        "post_decline_entry_share": _share_from(incidence, onset + horizon if onset is not None else None),
        "post_first_cohort_entry_share": _share_from(incidence, first_entry + horizon if first_entry else None),
        "cohort_persistence_drop": drop,
        "post_peak_entry_share": _share_from(incidence, peak + 1 if peak is not None else None),
        "attribute_selection_trend": _spearman(entry_steps, fits),
        "outcome_selection_trend": _spearman([e for e, _p in persisted], [p for _e, p in persisted]),
        "interior_entry_peak": bool(peak is not None and first_entry is not None and peak > first_entry),
    }


# -- runs --------------------------------------------------------------------------------


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    clock_id: str,
    visibility_id: str,
    world_id: str,
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    schedule = clock_schedules(contract, profile)[clock_id]
    horizon = int(contract["h3_adapter"]["cohort_persistence_horizon_steps"])
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows = []
    for arm_id, evidence_regime in MAIN_ARMS:
        adapter_class = surge_adapter(schedule)
        with adapter_slot(adapter_class):
            run = rdc.run_arm(
                contract, config, replicate=replicate, visibility_id=visibility_id, world_id=world_id,
                arm_id=arm_id, evidence_regime=evidence_regime, probe=probe, q=1.0,
            )
        if rdc.HazardCommitAdapter is not ORIGINAL_ADAPTER:
            raise RuntimeError("the adapter slot was not restored")
        if len(adapter_class.created) != 1:
            raise RuntimeError("the surge adapter was not used exactly once")
        rows.append(
            {
                "replicate": replicate,
                "clock_id": clock_id,
                "visibility_id": visibility_id,
                "world_id": world_id,
                "arm_id": arm_id,
                "evidence_regime": evidence_regime,
                **run,
                **run_observables(run, adapter_class.created[0], horizon=horizon, steps=config.steps),
            }
        )
    return rows


def check_nesting(contract: Mapping[str, Any], profile: str, replicate: int) -> None:
    """A flat clock reproduces the N01 q = 1 arms exactly, which nest their predecessors."""

    config = abm1.study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, "linear_fit")
    flat = surge_adapter((1.0,) * (config.steps - config.burn_in_steps))
    nested_runs = {}
    for arm_id, evidence_regime in (MAIN_ARMS[0], MAIN_ARMS[3]):
        kwargs = {
            "replicate": replicate, "visibility_id": "immediate", "world_id": "linear_fit",
            "arm_id": arm_id, "evidence_regime": evidence_regime, "probe": probe, "q": 1.0,
        }
        reference = rdc.run_arm(contract, config, **kwargs)
        with adapter_slot(flat):
            nested = rdc.run_arm(contract, config, **kwargs)
        for key, value in reference.items():
            if nested[key] != value:
                raise RuntimeError(f"a flat surge clock does not nest the N01 arm at {key}")
        nested_runs[arm_id] = nested
    rdc.check_nesting(
        contract, config, replicate=replicate, visibility_id="immediate", runs=nested_runs, probe=probe
    )


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_study(
    contract: Mapping[str, Any],
    profile: str,
    workers: int = 1,
    clocks: Sequence[str] | None = None,
    visibilities: Sequence[str] = VISIBILITY_IDS,
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    clock_ids = list(clocks) if clocks is not None else [row["id"] for row in contract["surge_clock"]["clocks"]]
    tasks = [
        (contract, profile, replicate, clock_id, visibility_id, world_id)
        for replicate in range(config.replicates)
        for clock_id in clock_ids
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


# -- decisions ----------------------------------------------------------------------------


def cell_status(indexed, replicates, contract, clock_id, visibility_id) -> dict[str, Any]:  # noqa: ANN001
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    minimum = float(decision["entry_channel"]["improvement_at_least"])

    def values(world: str, arm: str, metric: str) -> np.ndarray:
        return np.asarray([float(indexed[(r, clock_id, visibility_id, world, arm)][metric]) for r in replicates])

    share = "productive_share_of_entrants"
    reports = {}
    for arm in NONLINEAR_ARMS:
        linear_gap = values("linear_fit", arm, share) - values("linear_fit", "linear_batch", share)
        for world in INTERACTION_FIT_WORLDS:
            world_gap = values(world, arm, share) - values(world, "linear_batch", share)
            reports[f"{arm}/{world}"] = abm1._judge(world_gap - linear_gap, minimum, required)
    saturated = all(
        int(np.sum(values(world, arm, "cumulative_entry_rate") >= SATURATION_ENTRY)) >= required
        for world in FIT_WORLDS
        for arm in FEATURE_ARMS
    )
    passed = sorted(key for key, report in reports.items() if report["pass"])
    status = "ENTRY_CHANNEL_ACTIVE" if passed else ("ENTRY_SATURATED" if saturated else "ENTRY_CHANNEL_ABSENT")
    return {"status": status, "passing_family_geometries": passed, "entry_channel_reports": reports}


def _mean(values: Sequence[Any]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return float(np.mean(kept)) if kept else None


def _accuracy(predicted: Sequence[bool], actual: Sequence[bool], tiers: Mapping[str, Any]) -> dict[str, Any]:
    correct = sum(p == a for p, a in zip(predicted, actual, strict=True))
    share = correct / len(actual)
    if correct == len(actual):
        disposition = "CONFIRMED_IN_ALL_CELLS"
    elif share >= float(tiers["mostly_holds_at_least"]):
        disposition = "HOLDS_IN_MOST_CELLS"
    else:
        disposition = "FAILS"
    return {
        "correct": correct,
        "cells": len(actual),
        "accuracy": share,
        "false_active": sum(p and not a for p, a in zip(predicted, actual, strict=True)),
        "false_inactive": sum(a and not p for p, a in zip(predicted, actual, strict=True)),
        "disposition": disposition,
    }


def separation(values: Sequence[float | None], active: Sequence[bool]) -> dict[str, Any]:
    pairs = [(v, a) for v, a in zip(values, active, strict=True) if v is not None]
    act = [v for v, a in pairs if a]
    ina = [v for v, a in pairs if not a]
    if not act or not ina:
        return {"disposition": "NOT_DISCRIMINATED", "active_range": None, "inactive_range": None}
    high = min(act) > max(ina)
    low = max(act) < min(ina)
    return {
        "disposition": "SEPARATES" if high or low else "DOES_NOT_SEPARATE",
        "direction": "ACTIVE_HIGHER" if high else ("ACTIVE_LOWER" if low else None),
        "active_range": [min(act), max(act)],
        "inactive_range": [min(ina), max(ina)],
        "missing_cells": len(values) - len(pairs),
    }


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(r["replicate"]), r["clock_id"], r["visibility_id"], r["world_id"], r["arm_id"]): r for r in rows
    }
    replicates = sorted({int(r["replicate"]) for r in rows})
    delays = {row["id"]: int(row["failure_delay_steps"]) for row in contract["visibility_regimes"]}
    discriminants = contract["discriminants"]
    cells = []
    for clock in contract["surge_clock"]["clocks"]:
        for visibility in VISIBILITY_IDS:
            feature = [
                r for r in rows
                if r["clock_id"] == clock["id"] and r["visibility_id"] == visibility and r["arm_id"] in FEATURE_ARMS
            ]
            width = _mean([r["entry_wave_width"] for r in feature])
            cell = {
                "clock_id": clock["id"],
                "visibility_id": visibility,
                "delay": delays[visibility],
                **cell_status(indexed, replicates, contract, clock["id"], visibility),
                "feature_arm_entry_wave_width": width,
                "feature_arm_relative_delay": (delays[visibility] + 1) / width if width else None,
                "post_failure_entry_share": _mean([r["post_failure_entry_share"] for r in feature]),
                "interior_peak_share": _mean([float(r["interior_entry_peak"]) for r in feature]),
                "model_h3_gap_mean": _mean([r["model_h3_gap"] for r in feature]),
                **{name: _mean([r[name] for r in feature]) for name in OBSERVABLES},
            }
            cells.append(cell)
    active = [cell["status"] == "ENTRY_CHANNEL_ACTIVE" for cell in cells]
    tiers = discriminants["accuracy_tiers"]
    pfes_threshold = float(discriminants["post_failure_entry_share"]["threshold"])
    wave_threshold = float(discriminants["feature_arm_relative_delay"]["threshold"])
    pfes = [cell["post_failure_entry_share"] for cell in cells]
    screen = {}
    for name in OBSERVABLES:
        values = [cell[name] for cell in cells]
        kept = [(v, p) for v, p in zip(values, pfes, strict=True) if v is not None]
        screen[name] = {
            **separation(values, active),
            "spearman_with_post_failure_entry_share": (
                _spearman([v for v, _p in kept], [p for _v, p in kept]) if len(kept) >= 3 else None
            ),
        }
    interior = _mean([cell["interior_peak_share"] for cell in cells])
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "cells": cells,
        "status_mix": {
            "active_cells": sum(active),
            "inactive_cells": len(active) - sum(active),
            "discriminating": 0 < sum(active) < len(active),
        },
        "design_check": {
            "mean_interior_peak_share": interior,
            "disposition": (
                "SURGE_CLOCK_PRODUCES_INTERIOR_PEAKS"
                if interior is not None and interior >= float(contract["decision_contract"]["interior_peak_share_at_least"])
                else "SURGE_CLOCK_DOES_NOT_PRODUCE_INTERIOR_PEAKS"
            ),
        },
        "post_failure_entry_share_test": {
            "threshold": pfes_threshold,
            **_accuracy([value is not None and value >= pfes_threshold for value in pfes], active, tiers),
            "separation_in_this_study": separation(pfes, active),
        },
        "feature_arm_relative_delay_test": {
            "threshold": wave_threshold,
            **_accuracy(
                [cell["feature_arm_relative_delay"] is not None and cell["feature_arm_relative_delay"] <= wave_threshold for cell in cells],
                active,
                tiers,
            ),
        },
        "observable_screen": screen,
        "separating_observables": sorted(name for name, entry in screen.items() if entry["disposition"] == "SEPARATES"),
    }


def summarize(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    config = abm1.study_config(contract, profile)
    metrics = ("cumulative_entry_rate", "productive_share_of_entrants", "probe_brier", "entry_wave_width", "post_failure_entry_share")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["clock_id"], row["visibility_id"], row["world_id"], row["arm_id"])].append(row)
    cells = []
    for (clock, visibility, world, arm), cell_rows in sorted(grouped.items()):
        cells.append(
            {
                "clock_id": clock, "visibility_id": visibility, "world_id": world, "arm_id": arm,
                "replicates": len(cell_rows),
                "means": {metric: _mean([row[metric] for row in cell_rows]) for metric in metrics},
            }
        )
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    return {
        "schema_version": "pivot_observable_discriminant_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "schedules": {key: [round(v, 6) for v in value] for key, value in clock_schedules(contract, profile).items()},
        "cells": cells,
        "abm_runs": len(rows),
        "dispositions": {"status": "NOT_EVALUATED_SMOKE_PROFILE"} if config.replicates < required else decide(rows, contract),
        "boundaries": contract["boundaries"],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Pivot observable-discriminant scorecard v1", "", f"Profile: `{summary['profile']}`", ""]
    dispositions = summary["dispositions"]
    if "cells" in dispositions:
        lines.extend(
            [
                "| Clock | Visibility | Status | PFES | Rel. delay | Interior peaks | " + " | ".join(OBSERVABLES) + " |",
                "|---|---|---|---:|---:|---:|" + "---:|" * len(OBSERVABLES),
            ]
        )
        for cell in dispositions["cells"]:
            lines.append(
                f"| {cell['clock_id']} | {cell['visibility_id']} | {cell['status']} | "
                f"{_fmt(cell['post_failure_entry_share'])} | {_fmt(cell['feature_arm_relative_delay'])} | "
                f"{_fmt(cell['interior_peak_share'])} | " + " | ".join(_fmt(cell[name]) for name in OBSERVABLES) + " |"
            )
        lines.extend(
            [
                "",
                "## Frozen dispositions",
                "",
                f"- Post-failure entry share: `{dispositions['post_failure_entry_share_test']['disposition']}`",
                f"- Feature-arm relative delay: `{dispositions['feature_arm_relative_delay_test']['disposition']}`",
                f"- Separating observables: {dispositions['separating_observables'] or 'none'}",
                f"- Design check: `{dispositions['design_check']['disposition']}`",
            ]
        )
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


def write_outputs(
    *, output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
    precommit: Mapping[str, Any], contract_path: Path,
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
        "schema_version": "pivot_observable_discriminant_receipt.v1",
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
        "clocks": {key: [round(v, 4) for v in value] for key, value in clock_schedules(contract, "development").items()},
        "cells": len(contract["surge_clock"]["clocks"]) * len(VISIBILITY_IDS),
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
        check_nesting(contract, args.profile, replicate)
    rows = run_study(contract, args.profile, workers=args.workers)
    summary = {**summarize(rows, contract, args.profile), "precommit_sha": args.precommit_sha}
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(), rows=rows, summary=summary, precommit=precommit, contract_path=config_path
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
