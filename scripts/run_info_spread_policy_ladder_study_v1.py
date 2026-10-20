"""Information-spread policy ladder, study v1 (stage D4 runner for D3a).

Design: docs/research/info-spread-policy-ladder-study-v1.md
Contract: experiments/configs/info_spread_policy_ladder_study_v1.yaml

A development run needs the precommit SHA, which must be HEAD with clean
precommit paths and matching source bindings. A smoke run and a pilot run
skip that check and are never read for results. cells.jsonl and summary.json
carry no timing, so a reproduction run must match them byte for byte; timing
goes to progress.jsonl and receipt.json.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS needs this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
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

import power_info_spread_policy_ladder_v2 as power2  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_study_v1.yaml"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
ARMS = ("F0", "F1", "T0_1", "T0_4", "L0")
COMPARISONS = {"closed_loop": ("L0", "T0_1"), "best_table": ("L0", "T0_4"), "hand_rule": ("L0", "F1")}


# -- contract and precommit --------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    design = contract["design"]
    groups = [(r, f, g) for r in design["regimes"] for f in design["families"][r] for g in design["geometries"]]
    if len(groups) != design["groups"]:
        raise ValueError("group count does not match regimes x families x geometries")
    if set(design["study_geometries"]) | {design["null_geometry"]} != set(design["geometries"]):
        raise ValueError("geometries must be the study geometries plus the null")
    if tuple(contract["arms"]) != ARMS:
        raise ValueError("arms differ from the runner's")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    verified = []
    for binding in contract["source_bindings"]:
        actual = _sha256(ROOT / binding["path"])
        if actual != binding["sha256"]:
            raise RuntimeError(f"source binding mismatch for {binding['path']}: {actual} != {binding['sha256']}")
        verified.append({"path": binding["path"], "sha256": actual})
    return verified


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    if FULL_SHA.fullmatch(commit_sha) is None:
        raise RuntimeError("a development run needs a full 40-character lowercase SHA")
    if _git("rev-parse", "HEAD") != commit_sha:
        raise RuntimeError("the precommit SHA is not HEAD")
    paths = [str(p) for p in contract["precommit"]["precommit_paths"]]
    if _git("status", "--porcelain", "--untracked-files=no", "--", *paths):
        raise RuntimeError("precommit paths are dirty")
    tracked = set(_git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    if sorted(set(paths) - tracked):
        raise RuntimeError("precommit paths are missing from the commit")
    return {"commit_sha": commit_sha, "precommit_paths": paths, "source_bindings": verify_source_bindings(contract)}


# -- one cell ------------------------------------------------------------------------


def _family_index(family: str) -> int:
    return {"linear_exact": 6, "mlp": 2, "partial_linear_exact": 7, "partial_mlp": 5}[family]


def apply_learner_settings(contract: Mapping[str, Any]) -> None:
    learners = contract["learners"]
    spread.use_learner_settings(learners["base_settings"])
    spread.LEARNER_SETTINGS["linear_exact"] = {"l2": float(learners["pooled_linear_l2"])}
    if float(learners["agent_residual_l2"]) != spread.AGENT_RESIDUAL_WEIGHT_DECAY:
        raise RuntimeError("the agent residual l2 differs from the prototype's")


def _learner(contract: Mapping[str, Any], world: spread.World, family: str, geometry_index: int,
             cell: int, replicate: int, device: str) -> spread.Policy:
    config = world.config
    components = (int(contract["seeds"]["learner_tag"]), cell, ord(config.regime),
                  _family_index(family), geometry_index, replicate)
    prior = spread.hand_prior(config)
    observability = contract["world"]["observability"]
    one_row = bool(observability["one_row_per_agent_and_item"])
    if config.regime == "A":
        return spread.pooled_learning_policy(
            family, prior, seed_components=components, label_mode=observability["label_mode"],
            one_row_per_exposure=one_row, device=device,
        )
    return spread.partial_pooling_policy(
        family.removeprefix("partial_"), prior, config.agents, seed_components=components,
        label_mode=observability["label_mode"], device=device, agent_residual="exact",
        one_row_per_exposure=one_row,
    )


def run_cell(contract: Mapping[str, Any], profile: str, regime: str, family: str, geometry: str,
             cell: int, device: str) -> dict[str, Any]:
    sizes = contract["profiles"][profile]
    world_spec = contract["world"]
    design = contract["design"]
    world = spread.build_world(spread.WorldConfig(
        agents=int(sizes["agents"]), steps=int(sizes["steps"]), items_per_step=int(sizes["items_per_step"]),
        degree=int(world_spec["degree"]), rewire=float(world_spec["rewire"]),
        seeds_per_item=int(world_spec["seeds_per_item"]), delay=int(world_spec["delay"]),
        visibility=float(world_spec["visibility"]), epsilon=float(world_spec["epsilon"]),
        refit_every=int(world_spec["refit_every"]), regime=regime, geometry=geometry,
        world_seed=int(contract["seeds"]["world_seed_offset"]) + cell,
    ))
    geometry_index = list(design["geometries"]).index(geometry)
    donors = [
        spread.simulate(world, spread.build_stream(world, r),
                        _learner(contract, world, family, geometry_index, cell, r, device), r).snapshots
        for r in contract["seeds"]["donor_replicates"]
    ]
    prior = spread.hand_prior(world.config)
    rows = []
    for replicate in range(int(sizes["replicates"])):
        stream = spread.build_stream(world, replicate)
        arms = {
            "F0": spread.threshold_policy(),
            "F1": spread.prior_policy("F1", prior),
            "T0_1": spread.table_policy("T0_1", donors[:1], final_only=False),
            "T0_4": spread.table_policy("T0_4", donors, final_only=False),
            "L0": _learner(contract, world, family, geometry_index, cell, replicate, device),
        }
        metrics = {}
        for name, policy in arms.items():
            result = spread.simulate(world, stream, policy, replicate)
            metrics[name] = {
                "share_precision": result.behavior["share_precision"],
                "false_reach": result.behavior["false_reach"],
                "decisions": result.decisions,
            }
        rows.append({"replicate": replicate, "arms": metrics})
    return {"regime": regime, "family": family, "geometry": geometry, "cell": cell, "rows": rows}


def device_for(family: str) -> str:
    import torch

    return "cuda:0" if "mlp" in family and torch.cuda.is_available() else "cpu"


def _task(args: tuple[str, str, int, str, str, str, int]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    contract_path, profile, index, regime, family, geometry, cell = args
    contract = load_contract(Path(contract_path))
    apply_learner_settings(contract)
    device = device_for(family)
    started = time.perf_counter()
    result = run_cell(contract, profile, regime, family, geometry, cell, device)
    return index, result, {"device": device, "task_seconds": round(time.perf_counter() - started, 1)}


# -- the registered rule ---------------------------------------------------------------


def _group_rule(contract: Mapping[str, Any], cell_means: Sequence[float], key: Sequence[int]) -> dict[str, Any]:
    rule = contract["rule"]
    if float(rule["minimum_effect"]) != power2.MIN_GROUP_EFFECT or int(rule["bootstrap_resamples"]) != 2000:
        raise RuntimeError("the rule differs from the group rule tested in D2b and D2c")
    return power2.group_rule(cell_means, key)


def summarize(contract: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["regime"], cell["family"], cell["geometry"]), []).append(cell)
    table: dict[str, Any] = {}
    tag = int(contract["seeds"]["rule_tag"])
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        for c_index, (name, (left, right)) in enumerate(COMPARISONS.items()):
            means = [
                float(np.mean([r["arms"][left]["share_precision"] - r["arms"][right]["share_precision"] for r in c["rows"]]))
                for c in members
            ]
            entry[name] = _group_rule(contract, means, (tag, g_index, c_index))
        entry["mean_share_precision"] = {
            arm: round(float(np.mean([r["arms"][arm]["share_precision"] for c in members for r in c["rows"]])), 4)
            for arm in ARMS
        }
        entry["mean_false_reach"] = {
            arm: round(float(np.mean([r["arms"][arm]["false_reach"] for c in members for r in c["rows"]])), 2)
            for arm in ARMS
        }
        table[f"{regime}/{family}/{geometry}"] = entry
    return {"table": table, "dispositions": dispositions(contract, table)}


def dispositions(contract: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    design = contract["design"]
    thresholds = contract["dispositions"]
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    positive = {
        regime: sum(
            1 for k, v in study.items()
            if k.startswith(f"{regime}/") and v["closed_loop"]["material"] and v["closed_loop"]["mean_cell_delta"] > 0
        )
        for regime in design["regimes"]
    }
    per_regime_groups = len(study) // len(design["regimes"])
    if positive["B"] >= thresholds["b_at_least"] and positive["A"] <= thresholds["a_at_most"]:
        closed_loop = "AS_EXPECTED"
    elif positive["B"] <= thresholds["b_at_most_for_table_suffices"]:
        closed_loop = "TABLE_SUFFICES_EVEN_IN_B"
    elif positive["A"] >= thresholds["a_at_least_for_closed_loop_in_a"]:
        closed_loop = "CLOSED_LOOP_MATTERS_EVEN_IN_A"
    else:
        closed_loop = "MIXED"

    def best_table(entry: Mapping[str, Any]) -> str:
        rule = entry["best_table"]
        if not rule["material"]:
            return "INDISTINGUISHABLE"
        return "ADAPTIVE_BEATS_BEST_TABLE" if rule["mean_cell_delta"] > 0 else "BEST_TABLE_BEATS_ADAPTIVE"

    best = {k: best_table(v) for k, v in study.items()}
    null_violations = [
        k for k, v in table.items()
        if k.split("/")[2] == design["null_geometry"] and v["hand_rule"]["material"] and v["hand_rule"]["mean_cell_delta"] > 0
    ]
    return {
        "design_check": "DESIGN_DEFECT_IN_NULL_WORLD" if null_violations else "NULL_WORLD_PASSES",
        "null_violations": null_violations,
        "closed_loop": closed_loop,
        "closed_loop_material_positive_groups": positive,
        "study_groups_per_regime": per_regime_groups,
        "best_table_by_group": best,
        "best_table_counts": {
            regime: {label: sum(1 for k, v in best.items() if k.startswith(f"{regime}/") and v == label)
                     for label in ("ADAPTIVE_BEATS_BEST_TABLE", "BEST_TABLE_BEATS_ADAPTIVE", "INDISTINGUISHABLE")}
            for regime in design["regimes"]
        },
    }


# -- main ------------------------------------------------------------------------------


def _host() -> dict[str, Any]:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"node": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "gpu": gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


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
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    design = contract["design"]
    cells_per_group = args.pilot_cells or int(contract["profiles"][args.profile]["cells"])
    specs = [
        (r, f, g, cell)
        for r in design["regimes"] for f in design["families"][r] for g in design["geometries"]
        for cell in range(cells_per_group)
    ]
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
                    "group": f'{cell["regime"]}/{cell["family"]}/{cell["geometry"]}', "cell": cell["cell"],
                    **timing, "elapsed_seconds": round(time.perf_counter() - started, 1),
                }) + "\n")
    ordered = sorted(results.values(), key=lambda c: (c["regime"], c["family"], c["geometry"], c["cell"]))
    cells_path = args.output_dir / "cells.jsonl"
    cells_path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in ordered), encoding="utf-8")
    summary = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "cells_per_group": cells_per_group,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        **summarize(contract, ordered),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": _host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"dispositions": summary["dispositions"]["closed_loop"],
                      "design_check": summary["dispositions"]["design_check"], **receipt["outputs"]}))


if __name__ == "__main__":
    main()
