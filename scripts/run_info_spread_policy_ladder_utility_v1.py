"""Information-spread policy ladder, utility study v1 (O25).

Design: docs/research/info-spread-policy-ladder-utility-v1.md
Contract: experiments/configs/info_spread_policy_ladder_utility_v1.yaml

Every registered comparison so far was scored on share precision on
unverified items. Precision rewards selectivity, and it is not the agents'
objective. This study scores the same arms, in the same cells and runs, by
the agents' utility on unverified items. A true share gains 1, a false share
costs the sharer c, and a decline is worth 0. It then asks whether each
registered disposition still holds.

- Regime A (c = 1 for every agent): utility is (true - false) / agents.
  It is computed from the counts already recorded by study v2, D5, and O24,
  with no new run.
- Regime B (costs 0.5, 1, 2): every arm is rerun with
  info_spread_utility_measure_v1, which adds the cost of each false share.
  Each rerun row must reproduce the recorded counts, precision, reach, and
  decisions exactly (parity), or reading is blocked.

The arms are F1, T0_1, T0_4, and L0 (study v2), P0, X0, and L1 (D5), and P0u
and L1u (O24). A development run needs the precommit SHA; smoke and pilot runs
skip that check and are never read for results. cells.jsonl and summary.json
carry no timing.
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

import info_spread_utility_measure_v1 as utility  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_unclipped_prior_v1 as unclipped  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_utility_v1.yaml"
ARMS = ("F1", "T0_1", "T0_4", "L0", "P0", "X0", "L1", "P0u", "L1u")
LEARNER_ARMS = ("F1", "T0_1", "T0_4", "L0", "L1", "L1u")
FIXED_ARMS = ("P0", "X0", "P0u")
COMPARISONS = {
    "closed_loop": ("L0", "T0_1"), "best_table": ("L0", "T0_4"), "hand_rule": ("L0", "F1"),
    "llm_prior_in_learner": ("L1", "L0"), "nabm_vs_archetypes": ("L0", "X0"),
    "closed_loop_on_llm_prior": ("L1", "P0"), "llm_prior_vs_hand": ("P0", "F1"),
    "unclipped_prior_vs_hand": ("P0u", "F1"), "unclipped_prior_in_learner": ("L1u", "L0"),
    "closed_loop_on_unclipped_prior": ("L1u", "P0u"),
}
LABELS = ("POSITIVE", "NEGATIVE", "INDISTINGUISHABLE")
UTILITY = "unverified_utility_per_agent"
PARITY_KEYS = ("share_precision", "unverified_share_precision", "false_reach", "decisions", "by_state")


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


def contracts(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (v2.load_contract(ROOT / contract["inputs"]["study_v2"]["contract"]),
            llm.load_contract(ROOT / contract["inputs"]["llm_arms"]["contract"]),
            unclipped.load_contract(ROOT / contract["inputs"]["unclipped_prior"]["contract"]))


verify_source_bindings = v2.verify_source_bindings
verify_precommit = v2.verify_precommit
_sha256 = v2._sha256


# -- recorded inputs ------------------------------------------------------------------------


def _bound_cells(record_path: Path, cells_path: Path) -> list[dict[str, Any]]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not all(record["reproduction"]["byte_identical"].values()):
        raise RuntimeError(f"{record_path.name} did not reproduce byte for byte")
    if _sha256(cells_path) != record["reproduction"]["outputs"]["development"]["cells.jsonl"]:
        raise RuntimeError(f"the cells differ from {record_path.name}")
    return [json.loads(line) for line in cells_path.read_text(encoding="utf-8").splitlines()]


def load_recorded(contract: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[tuple[str, str, str, int], list]:
    """Recorded rows per (regime, family, geometry, cell): every arm's metrics by replicate."""

    inputs = contract["inputs"]
    v2_cells = _bound_cells(ROOT / inputs["study_v2"]["record"], paths["study_v2"])
    d5_cells = _bound_cells(ROOT / inputs["llm_arms"]["record"], paths["llm_arms"])
    o24_cells = _bound_cells(ROOT / inputs["unclipped_prior"]["record"], paths["unclipped_prior"])
    return join_recorded(v2_cells, d5_cells, o24_cells)


def join_recorded(v2_cells: Sequence[Mapping[str, Any]], d5_cells: Sequence[Mapping[str, Any]],
                  o24_cells: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], list]:
    fixed = {}
    learner = {}
    for source in (d5_cells, o24_cells):
        for c in source:
            if c["kind"] == "fixed":
                fixed.setdefault((c["regime"], c["geometry"], c["cell"]), []).append(c)
            else:
                learner.setdefault((c["regime"], c["family"], c["geometry"], c["cell"]), []).append(c)
    joined = {}
    for c in v2_cells:
        key = (c["regime"], c["family"], c["geometry"], c["cell"])
        sources = [c, *fixed[(c["regime"], c["geometry"], c["cell"])], *learner[key]]
        rows = []
        for parts in zip(*(s["rows"] for s in sources), strict=True):
            if len({p["replicate"] for p in parts}) != 1:
                raise RuntimeError("replicates are not aligned")
            merged: dict[str, Any] = {}
            for part in parts:
                merged.update({arm: metrics for arm, metrics in part["arms"].items() if arm in ARMS})
            rows.append({"replicate": parts[0]["replicate"], "arms": merged})
        joined[key] = rows
    return joined


# -- Regime B reruns ------------------------------------------------------------------------


def _measured(world: spread.World, stream: spread.Stream, policy: spread.Policy, replicate: int) -> dict[str, Any]:
    measured = utility.measured_simulate(world, stream, policy, replicate)[1]
    return {key: measured[key] for key in (*PARITY_KEYS, UTILITY)}


def priors_and_tables(llm_contract: Mapping[str, Any], log_odds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    arms = llm.build_llm_arms(llm_contract, log_odds)
    return {"P0": arms["P0"], "X0": arms["X0"], "P0u": unclipped.build_unclipped_priors(llm_contract, log_odds)["priors"]}


def run_learner_cell(v2_contract: Mapping[str, Any], llm_arms: Mapping[str, Any], profile: str, regime: str,
                     family: str, geometry: str, cell: int, device: str) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    gi = list(v2_contract["design"]["geometries"]).index(geometry)
    hand = spread.hand_prior(world.config)
    donors = [spread.simulate(world, spread.build_stream(world, r),
                              v2._learner(v2_contract, world, family, gi, cell, r, device), r).snapshots
              for r in v2_contract["seeds"]["donor_replicates"]]
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        arms: dict[str, Callable[[], spread.Policy]] = {
            "F1": lambda: spread.prior_policy("F1", hand),
            "T0_1": lambda: spread.table_policy("T0_1", donors[:1], final_only=False),
            "T0_4": lambda: spread.table_policy("T0_4", donors, final_only=False),
            "L0": lambda r=replicate: v2._learner(v2_contract, world, family, gi, cell, r, device),
            "L1": lambda r=replicate: llm._learner(v2_contract, world, family, gi, cell, r, device,
                                                   llm_arms["P0"][regime]),
            "L1u": lambda r=replicate: llm._learner(v2_contract, world, family, gi, cell, r, device,
                                                    llm_arms["P0u"][regime]),
        }
        rows.append({"replicate": replicate,
                     "arms": {name: _measured(world, stream, make(), replicate) for name, make in arms.items()}})
    return {"kind": "learner", "regime": regime, "family": family, "geometry": geometry, "cell": cell, "rows": rows}


def run_fixed_cell(llm_contract: Mapping[str, Any], v2_contract: Mapping[str, Any], llm_arms: Mapping[str, Any],
                   profile: str, regime: str, geometry: str, cell: int) -> dict[str, Any]:
    world = v2.build_world(v2_contract, profile, regime, geometry, cell)
    rows = []
    for replicate in range(int(v2_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        policies = {
            "P0": spread.prior_policy("P0", llm_arms["P0"][regime]),
            "X0": llm.x0_policy(llm_contract, world, stream, llm_arms["X0"][regime], []),
            "P0u": spread.prior_policy("P0u", llm_arms["P0u"][regime]),
        }
        rows.append({"replicate": replicate,
                     "arms": {name: _measured(world, stream, p, replicate) for name, p in policies.items()}})
    return {"kind": "fixed", "regime": regime, "geometry": geometry, "cell": cell, "rows": rows}


# -- the registered rule -----------------------------------------------------------------


def utility_rows(contract: Mapping[str, Any], v2_contract: Mapping[str, Any],
                 recorded: Mapping[tuple[str, str, str, int], list],
                 reruns: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[str, str, str, int], list], list[str]]:
    """Utility per row and arm: computed from counts in Regime A, rerun in Regime B with parity."""

    agents = int(v2_contract["profiles"][contract["profile_for_counts"]]["agents"])
    fixed = {(c["regime"], c["geometry"], c["cell"]): c for c in reruns if c["kind"] == "fixed"}
    learners = {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in reruns if c["kind"] == "learner"}
    out, parity_failures = {}, []
    for key, rows in recorded.items():
        regime, family, geometry, cell = key
        values = []
        for index, row in enumerate(rows):
            if regime == "A":
                values.append({arm: utility.utility_from_counts(m["by_state"], agents) for arm, m in row["arms"].items()})
                continue
            rerun = {**learners[key]["rows"][index]["arms"], **fixed[(regime, geometry, cell)]["rows"][index]["arms"]}
            for arm, metrics in row["arms"].items():
                if any(rerun[arm][k] != metrics[k] for k in PARITY_KEYS):
                    parity_failures.append(f"{regime}/{family}/{geometry}/{cell}/{row['replicate']}/{arm}")
            values.append({arm: rerun[arm][UTILITY] for arm in row["arms"]})
        out[key] = values
    return out, sorted(set(parity_failures))


def _label(rule: Mapping[str, Any]) -> str:
    if not rule["material"]:
        return "INDISTINGUISHABLE"
    return "POSITIVE" if rule["mean_cell_delta"] > 0 else "NEGATIVE"


def group_rule(values: Sequence[float], key: Sequence[int], minimum: float, resamples: int) -> dict[str, Any]:
    """The study's group rule, with a minimum effect in utility units."""

    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence(tuple(int(k) for k in key)))
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(resamples)]
    low, high = (float(q) for q in np.quantile(means, [0.025, 0.975]))
    mean = float(values.mean())
    return {"mean_cell_delta": round(mean, 4), "interval_95": [round(low, 4), round(high, 4)],
            "minimum_effect": round(minimum, 4), "material": bool((low > 0 or high < 0) and abs(mean) >= minimum)}


def summarize(contract: Mapping[str, Any], v2_contract: Mapping[str, Any],
              values: Mapping[tuple[str, str, str, int], list]) -> dict[str, Any]:
    design = v2_contract["design"]
    groups: dict[tuple[str, str, str], list[list[dict[str, float]]]] = {}
    for (regime, family, geometry, cell), rows in sorted(values.items()):
        groups.setdefault((regime, family, geometry), []).append(rows)
    fraction = float(contract["rule"]["minimum_effect_fraction_of_hand_rule"])
    minimum = {}
    for regime in design["regimes"]:
        f1 = [np.mean([r["F1"] for rows in members for r in rows]) for (rg, _f, g), members in groups.items()
              if rg == regime and g in design["study_geometries"]]
        minimum[regime] = fraction * float(np.mean(np.abs(f1)))
    tag, resamples = int(contract["seeds"]["rule_tag"]), int(contract["rule"]["bootstrap_resamples"])
    table: dict[str, Any] = {}
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        for c_index, (name, (left, right)) in enumerate(COMPARISONS.items()):
            means = [float(np.mean([r[left] - r[right] for r in rows])) for rows in members]
            entry[name] = group_rule(means, (tag, g_index, c_index), minimum[regime], resamples)
        entry["mean_utility"] = {arm: round(float(np.mean([r[arm] for rows in members for r in rows])), 3)
                                 for arm in ARMS}
        table[f"{regime}/{family}/{geometry}"] = entry
    return {"minimum_effect": {k: round(v, 4) for k, v in minimum.items()}, "table": table}


def label_counts(table: Mapping[str, Any], v2_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Labels per comparison and regime, over the study groups of a {group: {comparison: rule}} table."""

    design = v2_contract["design"]
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    counts = {}
    for name in COMPARISONS:
        counts[name] = {}
        for regime in design["regimes"]:
            labels = [_label(v[name]) for k, v in study.items() if k.startswith(f"{regime}/")]
            counts[name][regime] = {label: labels.count(label) for label in LABELS}
    return counts


def precision_tables(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Each comparison's recorded group rule on precision, from the three records."""

    inputs = contract["inputs"]
    names = {"study_v2": ("closed_loop", "best_table", "hand_rule"),
             "llm_arms": ("llm_prior_in_learner", "nabm_vs_archetypes", "closed_loop_on_llm_prior", "llm_prior_vs_hand"),
             "unclipped_prior": ("unclipped_prior_vs_hand", "unclipped_prior_in_learner",
                                 "closed_loop_on_unclipped_prior")}
    table: dict[str, dict[str, Any]] = {}
    for source, comparisons in names.items():
        record = json.loads((ROOT / inputs[source]["record"]).read_text(encoding="utf-8"))
        for group, entry in record["table"].items():
            for name in comparisons:
                table.setdefault(group, {})[name] = entry[v2.PRIMARY][name]
    return table


def dispositions(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], utility_table: Mapping[str, Any]) -> dict[str, Any]:
    at_least = int(contract["dispositions"]["mostly_at_least"])
    by_utility = label_counts(utility_table, v2_contract)
    by_precision = label_counts(precision_tables(contract), v2_contract)

    def summary(c: Mapping[str, int]) -> str:
        if c["POSITIVE"] >= at_least:
            return "MOSTLY_POSITIVE"
        if c["NEGATIVE"] >= at_least:
            return "MOSTLY_NEGATIVE"
        return "MIXED_OR_NONE"

    rows = {}
    for name in COMPARISONS:
        for regime in v2_contract["design"]["regimes"]:
            p, u = summary(by_precision[name][regime]), summary(by_utility[name][regime])
            rows[f"{name}/{regime}"] = {"precision": p, "utility": u, "agrees": p == u,
                                        "precision_counts": by_precision[name][regime],
                                        "utility_counts": by_utility[name][regime]}
    disagreements = sorted(k for k, v in rows.items() if not v["agrees"])
    # No null-world defect check here: in the null world every share has negative expected utility (base rate 0.4
    # below the cutoff), so a learner that rightly learns the base rate gains utility by sharing less. Null groups
    # are reported, not judged; reruns are validated by parity instead.
    return {"by_comparison": rows, "disagreements": disagreements,
            "overall": "CONCLUSIONS_HOLD_UNDER_UTILITY" if not disagreements else "SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY"}


# -- main --------------------------------------------------------------------------------


def _context(contract_path: str, elicitation_dir: str) -> tuple[dict, dict, dict, dict]:
    contract = load_contract(Path(contract_path))
    v2_contract, llm_contract, _o24 = contracts(contract)
    log_odds = unclipped.load_registered_elicitation(_o24, llm_contract, Path(elicitation_dir))
    return contract, v2_contract, llm_contract, priors_and_tables(llm_contract, log_odds)


def _learner_task(args: tuple[str, str, str, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, profile, regime, family, geometry, cell = args
    _c, v2_contract, _llm, arms = _context(contract_path, elicitation_dir)
    v2.apply_learner_settings(v2_contract)
    device = v2.device_for(family)
    started = time.perf_counter()
    result = run_learner_cell(v2_contract, arms, profile, regime, family, geometry, cell, device)
    return (f"learner/{regime}/{family}/{geometry}/{cell}", result,
            {"device": device, "task_seconds": round(time.perf_counter() - started, 1)})


def _fixed_task(args: tuple[str, str, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, profile, regime, geometry, cell = args
    _c, v2_contract, llm_contract, arms = _context(contract_path, elicitation_dir)
    started = time.perf_counter()
    result = run_fixed_cell(llm_contract, v2_contract, arms, profile, regime, geometry, cell)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    v2_contract, _llm, _o24 = contracts(contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    inputs = {"study_v2": args.v2_cells, "llm_arms": args.d5_cells, "unclipped_prior": args.o24_cells}
    if is_registered_run and any(p is None for p in inputs.values()):
        parser.error("a registered run needs --v2-cells, --d5-cells, and --o24-cells")
    design = v2_contract["design"]
    cells_per_group = args.pilot_cells or int(v2_contract["profiles"][args.profile]["cells"])
    base = (str(args.contract), str(args.elicitation_dir), args.profile)
    learner_specs = [("B", f, g, cell) for f in design["families"]["B"] for g in design["geometries"]
                     for cell in range(cells_per_group)]
    learner_specs.sort(key=lambda s: "mlp" not in s[1])  # heaviest first, for load balance
    fixed_specs = [("B", g, cell) for g in design["geometries"] for cell in range(cells_per_group)]
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
        "profile": args.profile, "cells_per_group": cells_per_group,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
    }
    if all(p is not None for p in inputs.values()):
        values, parity_failures = utility_rows(contract, v2_contract, load_recorded(contract, inputs), ordered)
        utility_summary = summarize(contract, v2_contract, values)
        summary.update({**utility_summary, "parity_failures": parity_failures,
                        "dispositions": dispositions(contract, v2_contract, utility_summary["table"])})
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": v2._host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"overall": summary.get("dispositions", {}).get("overall"),
                      "disagreements": summary.get("dispositions", {}).get("disagreements"),
                      "parity_failures": len(summary.get("parity_failures", [])), **receipt["outputs"]}))


if __name__ == "__main__":
    main()
