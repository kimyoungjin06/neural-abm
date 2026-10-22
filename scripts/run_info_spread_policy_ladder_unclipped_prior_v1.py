"""Information-spread policy ladder, unclipped-prior study v1 (O24).

Design: docs/research/info-spread-policy-ladder-unclipped-prior-v1.md
Contract: experiments/configs/info_spread_policy_ladder_unclipped_prior_v1.yaml

D5's once-elicited prior P0 distilled the model's log-odds after clipping
each answer's probability to [1e-4, 1 - 1e-4], a rule inherited from Pivot.
In Regime A, 99% of the answers fell below the clip, so P0/A came out nearly
constant. This study rebuilds the prior from the same registered elicitation
without the clip:

- P0u: the raw log-odds distilled by the same ridge regression and shifted
  to the same stated base rate;
- L1u: the study learner started from P0u.

No LLM is called. The elicitation is D5's first registered elicitation,
checked against the D3b prompts and bound by the D5 record. F1, L0, P0, and
L1 are read from the study v2 and D5 development cells, bound by their
records. At replicate 0 of every L1u task, L0 is recomputed and must equal
study v2's value.

A development run needs the precommit SHA; smoke and pilot runs skip that
check and are never read for results. cells.jsonl and summary.json carry no
timing.
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
import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_unclipped_prior_v1.yaml"
ARMS = ("P0u", "L1u")
COMPARISONS = {
    "unclipped_prior_vs_hand": ("P0u", "F1"),
    "unclipped_prior_in_learner": ("L1u", "L0"),
    "closed_loop_on_unclipped_prior": ("L1u", "P0u"),
}
DESCRIPTIVE = {"readout_effect_on_the_prior": ("P0u", "P0"), "readout_effect_on_the_learner": ("L1u", "L1")}
LABELS = ("POSITIVE", "NEGATIVE", "INDISTINGUISHABLE")


# -- contract --------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    if tuple(contract["arms"]) != ARMS:
        raise ValueError("arms differ from the runner's")
    if {k: tuple(v) for k, v in contract["comparisons"].items()} != COMPARISONS:
        raise ValueError("comparisons differ from the runner's")
    return contract


def load_llm_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return llm.load_contract(ROOT / contract["llm_arms"]["contract"])


def load_v2_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return v2.load_contract(ROOT / contract["study_v2"]["contract"])


verify_source_bindings = v2.verify_source_bindings
verify_precommit = v2.verify_precommit
_sha256 = v2._sha256


# -- the unclipped prior --------------------------------------------------------------------


def build_unclipped_priors(llm_contract: Mapping[str, Any], log_odds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """P0u per regime: D5's distillation and calibration on the raw log-odds."""

    calibration = llm_contract["calibration"]
    priors: dict[str, core.LinearPrior] = {}
    report: dict[str, Any] = {}
    for regime in ("A", "B"):
        features, _contacts = llm.p0_cohort(llm_contract, regime)
        raw = np.asarray(log_odds[f"P0/{regime}"], dtype=np.float64)
        weights, bias, r_squared = llm.distill(features, raw, float(llm_contract["p0"]["ridge_per_profile"]))
        shift = llm.shift_to_base_rate(features @ weights + bias, float(calibration["base_rate"]), calibration)
        priors[regime] = core.LinearPrior(weights, bias + shift)
        report[regime] = {"p0u_weights": [round(float(w), 6) for w in weights], "p0u_bias": round(bias, 6),
                          "p0u_calibration_shift": round(shift, 6), "p0u_distillation_r_squared": round(r_squared, 6)}
    return {"priors": priors, "report": report}


def load_registered_elicitation(contract: Mapping[str, Any], llm_contract: Mapping[str, Any],
                                directory: Path) -> dict[str, np.ndarray]:
    """D5's first elicitation, bound by the D5 record."""

    record = json.loads((ROOT / contract["llm_arms"]["record"]).read_text(encoding="utf-8"))
    if _sha256(directory / "elicitation.jsonl") != record["elicitation"]["sha256"]["first"]:
        raise RuntimeError("this is not D5's registered first elicitation")
    return llm.load_elicitation(directory, llm_contract)


# -- cells -------------------------------------------------------------------------------


def _measured(world: spread.World, stream: spread.Stream, policy: spread.Policy, replicate: int) -> dict[str, Any]:
    return v2.metrics(measure.measured_simulate(world, stream, policy, replicate)[1])


def run_fixed_cell(v2_contract: Mapping[str, Any], priors: Mapping[str, core.LinearPrior], profile: str,
                   regime: str, geometry: str, cell: int) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        rows.append({"replicate": replicate,
                     "arms": {"P0u": _measured(world, stream, spread.prior_policy("P0u", priors[regime]), replicate)}})
    return {"kind": "fixed", "regime": regime, "geometry": geometry, "cell": cell, "rows": rows}


def run_learner_cell(v2_contract: Mapping[str, Any], priors: Mapping[str, core.LinearPrior], profile: str,
                     regime: str, family: str, geometry: str, cell: int, device: str) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    geometry_index = list(v2_contract["design"]["geometries"]).index(geometry)
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        policy = llm._learner(v2_contract, world, family, geometry_index, cell, replicate, device, priors[regime])
        rows.append({"replicate": replicate, "arms": {"L1u": _measured(world, stream, policy, replicate)}})
    stream = spread.build_stream(world, 0)
    parity = _measured(world, stream, llm._learner(v2_contract, world, family, geometry_index, cell, 0, device,
                                                   spread.hand_prior(world.config)), 0)
    return {"kind": "learner", "regime": regime, "family": family, "geometry": geometry, "cell": cell,
            "rows": rows, "l0_replicate_0": parity}


# -- recorded inputs ------------------------------------------------------------------------


def _bound_cells(record_path: Path, cells_path: Path) -> list[dict[str, Any]]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not all(record["reproduction"]["byte_identical"].values()):
        raise RuntimeError(f"{record_path.name} did not reproduce byte for byte")
    if _sha256(cells_path) != record["reproduction"]["outputs"]["development"]["cells.jsonl"]:
        raise RuntimeError(f"the cells differ from {record_path.name}")
    return [json.loads(line) for line in cells_path.read_text(encoding="utf-8").splitlines()]


def load_recorded_cells(contract: Mapping[str, Any], v2_cells: Path, d5_cells: Path) -> dict[str, Any]:
    v2_rows = _bound_cells(ROOT / contract["study_v2"]["record"], v2_cells)
    d5_rows = _bound_cells(ROOT / contract["llm_arms"]["record"], d5_cells)
    return {
        "v2": {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in v2_rows},
        "d5_fixed": {(c["regime"], c["geometry"], c["cell"]): c for c in d5_rows if c["kind"] == "fixed"},
        "d5_learner": {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in d5_rows if c["kind"] == "learner"},
    }


# -- the registered rule -----------------------------------------------------------------


def _label(rule: Mapping[str, Any]) -> str:
    if not rule["material"]:
        return "INDISTINGUISHABLE"
    return "POSITIVE" if rule["mean_cell_delta"] > 0 else "NEGATIVE"


def summarize(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], recorded: Mapping[str, Any],
              cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixed = {(c["regime"], c["geometry"], c["cell"]): c for c in cells if c["kind"] == "fixed"}
    learners = {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in cells if c["kind"] == "learner"}
    parity_failures = [f"{r}/{f}/{g}/{cell}" for (r, f, g, cell), c in sorted(learners.items())
                       if c["l0_replicate_0"] != recorded["v2"][(r, f, g, cell)]["rows"][0]["arms"]["L0"]]
    groups: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = {}
    for (regime, family, geometry, cell), learner in sorted(learners.items()):
        sources = (recorded["v2"][(regime, family, geometry, cell)], recorded["d5_fixed"][(regime, geometry, cell)],
                   recorded["d5_learner"][(regime, family, geometry, cell)], fixed[(regime, geometry, cell)], learner)
        rows = []
        for parts in zip(*(s["rows"] for s in sources), strict=True):
            if len({p["replicate"] for p in parts}) != 1:
                raise RuntimeError("replicates are not aligned")
            merged: dict[str, Any] = {}
            for part in parts:
                merged.update(part["arms"])
            rows.append({"replicate": parts[0]["replicate"], "arms": merged})
        groups.setdefault((regime, family, geometry), []).append(rows)
    tag = int(contract["seeds"]["rule_tag"])
    every = {**COMPARISONS, **DESCRIPTIVE}
    arms = ("F1", "L0", "P0", "L1", *ARMS)
    table: dict[str, Any] = {}
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        for m_index, name in enumerate(v2.MEASURES):
            rules = {}
            for c_index, (comparison, (left, right)) in enumerate(every.items()):
                means = [float(np.mean([r["arms"][left][name] - r["arms"][right][name] for r in rows]))
                         for rows in members]
                if not np.all(np.isfinite(means)):
                    raise RuntimeError(f"a {name} is undefined in {regime}/{family}/{geometry}")
                rules[comparison] = v2._group_rule(contract, means, (tag, g_index, m_index, c_index))
            entry[name] = rules
            entry[f"mean_{name}"] = {
                arm: round(float(np.mean([r["arms"][arm][name] for rows in members for r in rows])), 4) for arm in arms
            }
        table[f"{regime}/{family}/{geometry}"] = entry
    primary = {key: entry[v2.PRIMARY] for key, entry in table.items()}
    return {"table": table, "parity_failures": parity_failures,
            "dispositions": dispositions(contract, v2_contract, primary)}


def dispositions(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    design = v2_contract["design"]
    at_least = int(contract["dispositions"]["mostly_at_least"])
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    labels = {name: {k: _label(v[name]) for k, v in study.items()} for name in COMPARISONS}
    counts = {name: {regime: {label: sum(1 for k, v in by_group.items() if k.startswith(f"{regime}/") and v == label)
                              for label in LABELS} for regime in design["regimes"]}
              for name, by_group in labels.items()}
    summary = {name: {regime: ("MOSTLY_POSITIVE" if c["POSITIVE"] >= at_least
                               else "MOSTLY_NEGATIVE" if c["NEGATIVE"] >= at_least else "MIXED_OR_NONE")
                      for regime, c in by_regime.items()}
               for name, by_regime in counts.items()}
    null_violations = [k for k, v in table.items() if k.split("/")[2] == design["null_geometry"]
                       and v["closed_loop_on_unclipped_prior"]["material"]
                       and v["closed_loop_on_unclipped_prior"]["mean_cell_delta"] > 0]
    return {"design_check": "DESIGN_DEFECT_IN_NULL_WORLD" if null_violations else "NULL_WORLD_PASSES",
            "null_violations": null_violations, "by_comparison": summary, "counts": counts, "labels_by_group": labels}


# -- main --------------------------------------------------------------------------------


def _context(contract_path: str, elicitation_dir: str) -> tuple[dict, dict, dict]:
    contract = load_contract(Path(contract_path))
    llm_contract = load_llm_contract(contract)
    v2_contract = load_v2_contract(contract)
    priors = build_unclipped_priors(llm_contract, load_registered_elicitation(contract, llm_contract,
                                                                              Path(elicitation_dir)))["priors"]
    return contract, v2_contract, priors


def _fixed_task(args: tuple[str, str, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, profile, regime, geometry, cell = args
    _contract, v2_contract, priors = _context(contract_path, elicitation_dir)
    started = time.perf_counter()
    result = run_fixed_cell(v2_contract, priors, profile, regime, geometry, cell)
    return f"fixed/{regime}/{geometry}/{cell}", result, {"device": "cpu",
                                                          "task_seconds": round(time.perf_counter() - started, 1)}


def _learner_task(args: tuple[str, str, str, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, profile, regime, family, geometry, cell = args
    _contract, v2_contract, priors = _context(contract_path, elicitation_dir)
    v2.apply_learner_settings(v2_contract)
    device = v2.device_for(family)
    started = time.perf_counter()
    result = run_learner_cell(v2_contract, priors, profile, regime, family, geometry, cell, device)
    return (f"learner/{regime}/{family}/{geometry}/{cell}", result,
            {"device": device, "task_seconds": round(time.perf_counter() - started, 1)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", choices=("development", "smoke"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--pilot-cells", type=int, default=None,
                        help="a full-load pilot with this many cells per group; never read for results")
    parser.add_argument("--elicitation-dir", type=Path, required=True, help="D5's first registered elicitation")
    parser.add_argument("--v2-cells", type=Path, help="study v2's development cells.jsonl")
    parser.add_argument("--d5-cells", type=Path, help="D5's development cells.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    llm_contract = load_llm_contract(contract)
    v2_contract = load_v2_contract(contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    if is_registered_run and (args.v2_cells is None or args.d5_cells is None):
        parser.error("a registered run needs --v2-cells and --d5-cells")
    report = build_unclipped_priors(llm_contract, load_registered_elicitation(contract, llm_contract,
                                                                              args.elicitation_dir))["report"]
    design = v2_contract["design"]
    cells_per_group = args.pilot_cells or int(v2_contract["profiles"][args.profile]["cells"])
    base = (str(args.contract), str(args.elicitation_dir), args.profile)
    learner_specs = [(r, f, g, cell) for r in design["regimes"] for f in design["families"][r]
                     for g in design["geometries"] for cell in range(cells_per_group)]
    learner_specs.sort(key=lambda s: "mlp" not in s[1])  # heaviest first, for load balance
    fixed_specs = [(r, g, cell) for r in design["regimes"] for g in design["geometries"] for cell in range(cells_per_group)]
    jobs: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [(_learner_task, (*base, *s)) for s in learner_specs]
    jobs += [(_fixed_task, (*base, *s)) for s in fixed_specs]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
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
        "profile": args.profile, "cells_per_group": cells_per_group, "primary_measure": v2.PRIMARY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        "elicitation_sha256": _sha256(args.elicitation_dir / "elicitation.jsonl"), "unclipped_priors": report,
    }
    if args.v2_cells is not None and args.d5_cells is not None:
        summary.update(summarize(contract, v2_contract, load_recorded_cells(contract, args.v2_cells, args.d5_cells),
                                 ordered))
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": v2._host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"dispositions": summary.get("dispositions", {}).get("by_comparison"),
                      "design_check": summary.get("dispositions", {}).get("design_check"),
                      "parity_failures": summary.get("parity_failures"), **receipt["outputs"]}))


if __name__ == "__main__":
    main()
