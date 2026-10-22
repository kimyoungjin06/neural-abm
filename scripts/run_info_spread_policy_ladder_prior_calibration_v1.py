"""Information-spread policy ladder, prior-calibration study v1 (O28).

Design: docs/research/info-spread-policy-ladder-prior-calibration-v1.md
Contract: experiments/configs/info_spread_policy_ladder_prior_calibration_v1.yaml

O25 found that, scored by the agents' utility, starting the learner from an
LLM prior instead of the hand rule F1 lowers utility: L1 - L0 in both regimes,
and L1u - L0 in Regime B. Every LLM prior was calibrated by D3b's rule, one
intercept shift to the base rate 0.4 over the design cohort, while F1 sits
near 0.68 there. This study asks whether the cost comes from that calibration
or from the model's answers.

A linear prior's logit over the design cohort has a location (its mean
probability) and a steepness (its standard deviation). Recalibrating prior D
to prior C keeps D's direction (its weights up to a positive scale) and takes
C's location and steepness:

    weight = D.weight * sd(cohort @ C.weight) / sd(cohort @ D.weight)
    bias   = the shift that makes the mean probability over the cohort C's

Crossing direction and calibration for each LLM prior Q (P0 and P0u) against
F1 gives a 2 x 2 per regime. Two corners are recorded (L0 from study v2, L1
and L1u from D5 and O24), and two are new learner arms:

- L1_hcal / L1u_hcal: Q's answers with F1's calibration;
- L0_pcal / L0_ucal: F1's answers with Q's calibration.

The fixed priors of the new corners run too, as a check that recalibration
on the cohort moves how much a prior shares in the world.

Utility per agent on unverified items is the measure (O25's). Regime A
recorded utility comes from the recorded counts and Regime B's from O25's
rerun cells, all bound by their records. At replicate 0 of every learner task,
L0 is recomputed and must equal its record. A development run needs the
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
import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_unclipped_prior_v1 as unclipped  # noqa: E402
import run_info_spread_policy_ladder_utility_v1 as o25  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_prior_calibration_v1.yaml"
# Each new prior: (the prior whose direction it keeps, the prior whose calibration it takes).
PRIORS = {"P0_hcal": ("P0", "F1"), "F1_pcal": ("F1", "P0"), "P0u_hcal": ("P0u", "F1"), "F1_ucal": ("F1", "P0u")}
LEARNERS = {"L1_hcal": "P0_hcal", "L0_pcal": "F1_pcal", "L1u_hcal": "P0u_hcal", "L0_ucal": "F1_ucal"}
RECORDED_LEARNERS = {"F1": "L0", "P0": "L1", "P0u": "L1u"}
COMPARISONS = {
    "P0": {
        "calibration_given_llm_answers": ("L1", "L1_hcal"),
        "calibration_given_hand_answers": ("L0_pcal", "L0"),
        "answers_given_hand_calibration": ("L1_hcal", "L0"),
        "answers_given_llm_calibration": ("L1", "L0_pcal"),
    },
    "P0u": {
        "calibration_given_llm_answers": ("L1u", "L1u_hcal"),
        "calibration_given_hand_answers": ("L0_ucal", "L0"),
        "answers_given_hand_calibration": ("L1u_hcal", "L0"),
        "answers_given_llm_calibration": ("L1u", "L0_ucal"),
    },
}
TOTALS = {"P0": ("L1", "L0", "llm_prior_in_learner"), "P0u": ("L1u", "L0", "unclipped_prior_in_learner")}
CALIBRATION = ("calibration_given_llm_answers", "calibration_given_hand_answers")
ANSWERS = ("answers_given_hand_calibration", "answers_given_llm_calibration")
LABELS = ("POSITIVE", "NEGATIVE", "INDISTINGUISHABLE")
UTILITY = o25.UTILITY
PARITY_KEYS = o25.PARITY_KEYS


def flat_comparisons() -> dict[str, tuple[str, str]]:
    return {f"{prior}/{name}": pair for prior, named in COMPARISONS.items() for name, pair in named.items()}


# -- contract --------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    if {k: tuple(v) for k, v in contract["priors"].items()} != PRIORS:
        raise ValueError("priors differ from the runner's")
    if contract["learner_arms"] != LEARNERS:
        raise ValueError("learner arms differ from the runner's")
    if {k: tuple(v) for k, v in contract["comparisons"].items()} != flat_comparisons():
        raise ValueError("comparisons differ from the runner's")
    return contract


def contracts(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs = contract["inputs"]
    return (v2.load_contract(ROOT / inputs["study_v2"]["contract"]),
            llm.load_contract(ROOT / inputs["llm_arms"]["contract"]),
            unclipped.load_contract(ROOT / inputs["unclipped_prior"]["contract"]))


verify_source_bindings = v2.verify_source_bindings
verify_precommit = v2.verify_precommit
_sha256 = v2._sha256


# -- priors ----------------------------------------------------------------------------


def cohort_calibration(prior: core.LinearPrior, cohort: np.ndarray) -> dict[str, float]:
    """A prior's location and steepness over the design cohort, and its share at the c = 1 cutoff."""

    logits = prior.logit(cohort)
    probability = core.sigmoid(logits)
    return {"mean_probability": float(probability.mean()), "logit_sd": float(np.std(logits)),
            "share_at_one_half": float(np.mean(probability >= 0.5))}


def recalibrate(direction: core.LinearPrior, source: core.LinearPrior, cohort: np.ndarray,
                calibration: Mapping[str, Any]) -> core.LinearPrior:
    """`direction`'s weights, rescaled to `source`'s logit sd and shifted to its mean probability."""

    weight = np.asarray(direction.weight, dtype=np.float64)
    spread_of_direction = float(np.std(cohort @ weight))
    if not spread_of_direction > 0:
        raise RuntimeError("a constant prior has no direction to keep")
    scaled = weight * (float(np.std(cohort @ np.asarray(source.weight, dtype=np.float64))) / spread_of_direction)
    target = float(core.sigmoid(source.logit(cohort)).mean())
    return core.LinearPrior(scaled, llm.shift_to_base_rate(cohort @ scaled, target, calibration))


def build_priors(llm_contract: Mapping[str, Any], log_odds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """F1, P0, and P0u as registered, and the four recalibrated priors, per regime."""

    p0 = llm.build_llm_arms(llm_contract, log_odds)["P0"]
    p0u = unclipped.build_unclipped_priors(llm_contract, log_odds)["priors"]
    priors: dict[str, dict[str, core.LinearPrior]] = {}
    report: dict[str, Any] = {}
    for regime in ("A", "B"):
        cohort, _contacts = llm.p0_cohort(llm_contract, regime)
        base = {"F1": spread.hand_prior(spread.WorldConfig(regime=regime)), "P0": p0[regime], "P0u": p0u[regime]}
        made = {name: recalibrate(base[d], base[c], cohort, llm_contract["calibration"])
                for name, (d, c) in PRIORS.items()}
        priors[regime] = {**base, **made}
        report[regime] = {
            name: {**{k: round(v, 6) for k, v in cohort_calibration(prior, cohort).items()},
                   "weights": [round(float(w), 6) for w in prior.weight], "bias": round(float(prior.bias), 6)}
            for name, prior in priors[regime].items()
        }
    return {"priors": priors, "report": report}


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


# -- recorded inputs ------------------------------------------------------------------------


def _shares(metrics: Mapping[str, Any]) -> int:
    return int(metrics["by_state"]["unrevealed"]["shares"] + metrics["by_state"]["pending"]["shares"])


def load_recorded(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], paths: Mapping[str, Path]) -> dict:
    inputs = contract["inputs"]
    cells = {name: o25._bound_cells(ROOT / inputs[name]["record"], paths[name])
             for name in ("study_v2", "llm_arms", "unclipped_prior", "utility")}
    return recorded_values(contract, v2_contract, cells)


def recorded_values(contract: Mapping[str, Any], v2_contract: Mapping[str, Any],
                    cells: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple, list]:
    """Per (regime, family, geometry, cell) and replicate: the recorded arms' metrics and utility.

    Regime A utility comes from the recorded counts; Regime B's from O25's rerun cells.
    """

    joined = o25.join_recorded(cells["study_v2"], cells["llm_arms"], cells["unclipped_prior"])
    agents = int(v2_contract["profiles"][contract["profile_for_counts"]]["agents"])
    reruns = {}
    for c in cells["utility"]:
        key = (c["regime"], c["geometry"], c["cell"]) if c["kind"] == "fixed" else \
            (c["regime"], c["family"], c["geometry"], c["cell"])
        reruns[key] = c
    out = {}
    for key, rows in joined.items():
        regime, _family, geometry, cell = key
        values = []
        for index, row in enumerate(rows):
            entry = {}
            for arm in ("F1", "P0", "P0u", "L0", "L1", "L1u"):
                metrics = row["arms"][arm]
                if regime == "A":
                    value = utility.utility_from_counts(metrics["by_state"], agents)
                else:
                    source = reruns[(regime, geometry, cell)] if arm in ("P0", "P0u") else reruns[key]
                    rerun = source["rows"][index]
                    if rerun["replicate"] != row["replicate"] or any(rerun["arms"][arm][k] != metrics[k]
                                                                       for k in PARITY_KEYS):
                        raise RuntimeError(f"O25's rerun differs from the record at {key}/{index}/{arm}")
                    value = rerun["arms"][arm][UTILITY]
                entry[arm] = {UTILITY: value, "unverified_shares": _shares(metrics), "parity": metrics,
                              v2.PRIMARY: metrics[v2.PRIMARY]}
            values.append({"replicate": row["replicate"], "arms": entry})
        out[key] = values
    return out


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


def parity_failures(recorded: Mapping[tuple, list], cells: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for c in cells:
        if c["kind"] != "learner":
            continue
        key = (c["regime"], c["family"], c["geometry"], c["cell"])
        old = recorded[key][0]["arms"]["L0"]
        new = c["l0_replicate_0"]
        if any(new[k] != old["parity"][k] for k in PARITY_KEYS) or new[UTILITY] != old[UTILITY]:
            failures.append("/".join(str(k) for k in key))
    return sorted(failures)


def calibration_check(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], recorded: Mapping[tuple, list],
                      cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does each recalibrated fixed prior share closer to its calibration source than to its direction source?"""

    design = v2_contract["design"]
    fixed = [c for c in cells if c["kind"] == "fixed" and c["geometry"] in design["study_geometries"]]
    result = {}
    for regime in design["regimes"]:
        first_family = design["families"][regime][0]
        shares = {arm: [] for arm in (*PRIORS, *RECORDED_LEARNERS)}
        for c in (c for c in fixed if c["regime"] == regime):
            old = recorded[(regime, first_family, c["geometry"], c["cell"])]
            for old_row, new_row in zip(old, c["rows"], strict=True):
                for arm in RECORDED_LEARNERS:
                    shares[arm].append(old_row["arms"][arm]["unverified_shares"])
                for arm in PRIORS:
                    shares[arm].append(_shares(new_row["arms"][arm]))
        means = {arm: float(np.mean(v)) for arm, v in shares.items()}
        for name, (direction, source) in PRIORS.items():
            result[f"{name}/{regime}"] = {
                "mean_unverified_shares": round(means[name], 1),
                "calibration_source": round(means[source], 1), "direction_source": round(means[direction], 1),
                "moves_to_its_calibration_source": abs(means[name] - means[source]) < abs(means[name] - means[direction]),
            }
    return result


def summarize(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], o25_record: Mapping[str, Any],
              recorded: Mapping[tuple, list], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    descriptive = learner_description(recorded, cells)
    check = calibration_check(contract, v2_contract, recorded, cells)
    return {"minimum_effect": o25_record["minimum_effect"], "table": table, "learner_description": descriptive,
            "calibration_check": check, "consistency_failures": consistency_failures,
            "parity_failures": parity_failures(recorded, cells),
            "dispositions": dispositions(contract, v2_contract, o25_record, table, check)}


def learner_description(recorded: Mapping[tuple, list], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean unverified shares and precision per learner arm and group; reported, not judged."""

    groups: dict[str, dict[str, dict[str, list[float]]]] = {}
    for c in (c for c in cells if c["kind"] == "learner"):
        key = (c["regime"], c["family"], c["geometry"], c["cell"])
        group = groups.setdefault(f"{c['regime']}/{c['family']}/{c['geometry']}", {})
        for old, new in zip(recorded[key], c["rows"], strict=True):
            for arm in RECORDED_LEARNERS.values():
                metrics = old["arms"][arm]
                group.setdefault(arm, {"shares": [], "precision": []})
                group[arm]["shares"].append(metrics["unverified_shares"])
                group[arm]["precision"].append(metrics[v2.PRIMARY])
            for arm, metrics in new["arms"].items():
                group.setdefault(arm, {"shares": [], "precision": []})
                group[arm]["shares"].append(_shares(metrics))
                group[arm]["precision"].append(metrics[v2.PRIMARY])
    return {g: {arm: {"mean_unverified_shares": round(float(np.mean(v["shares"])), 1),
                      "mean_unverified_share_precision": round(float(np.mean(v["precision"])), 4)}
                for arm, v in by_arm.items()}
            for g, by_arm in sorted(groups.items())}


def _summary_label(counts: Mapping[str, int], at_least: int) -> str:
    if counts["POSITIVE"] >= at_least:
        return "MOSTLY_POSITIVE"
    if counts["NEGATIVE"] >= at_least:
        return "MOSTLY_NEGATIVE"
    return "MIXED_OR_NONE"


def dispositions(contract: Mapping[str, Any], v2_contract: Mapping[str, Any], o25_record: Mapping[str, Any],
                 table: Mapping[str, Any], check: Mapping[str, Any]) -> dict[str, Any]:
    design = v2_contract["design"]
    at_least = int(contract["dispositions"]["mostly_at_least"])
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    by_case: dict[str, Any] = {}
    for prior, named in COMPARISONS.items():
        for regime in design["regimes"]:
            counts = {}
            for name in named:
                labels = [o25._label(v[f"{prior}/{name}"]) for k, v in study.items() if k.startswith(f"{regime}/")]
                counts[name] = {label: labels.count(label) for label in LABELS}
            summary = {name: _summary_label(c, at_least) for name, c in counts.items()}
            calibration = all(summary[n] == "MOSTLY_NEGATIVE" for n in CALIBRATION)
            answers = all(summary[n] == "MOSTLY_NEGATIVE" for n in ANSWERS)
            checked = all(check[f"{name}/{regime}"]["moves_to_its_calibration_source"]
                          for name, (d, c) in PRIORS.items() if prior in (d, c))
            attribution = ("CALIBRATION_CHECK_FAILED" if not checked else
                           {(True, False): "CALIBRATION", (False, True): "ANSWERS", (True, True): "BOTH",
                            (False, False): "NEITHER_CONSISTENTLY"}[(calibration, answers)])
            by_case[f"{prior}/{regime}"] = {"by_comparison": summary, "counts": counts, "attribution": attribution}
    by_o25 = o25_record["dispositions"]["by_comparison"]
    costed = [f"{prior}/{regime}" for prior, (_l, _r, name) in TOTALS.items() for regime in design["regimes"]
              if by_o25[f"{name}/{regime}"]["utility"] == "MOSTLY_NEGATIVE"]
    if costed != list(contract["dispositions"]["costed_cases"]):
        raise RuntimeError(f"O25's costed cases {costed} differ from the contract's")
    attributions = {by_case[case]["attribution"] for case in costed}
    if "CALIBRATION_CHECK_FAILED" in attributions:
        overall = "O28_CALIBRATION_CHECK_FAILED"
    elif len(attributions) == 1:
        overall = f"O28_{attributions.pop()}_IN_EVERY_COSTED_CASE"
    else:
        overall = "O28_ATTRIBUTION_DIFFERS_ACROSS_COSTED_CASES"
    return {"by_case": by_case, "costed_cases": costed, "overall": overall}


# -- main --------------------------------------------------------------------------------


def _context(contract_path: str, elicitation_dir: str) -> tuple[dict, dict, dict]:
    contract = load_contract(Path(contract_path))
    v2_contract, llm_contract, o24_contract = contracts(contract)
    log_odds = unclipped.load_registered_elicitation(o24_contract, llm_contract, Path(elicitation_dir))
    return contract, v2_contract, build_priors(llm_contract, log_odds)["priors"]


def _learner_task(args: tuple[str, str, str, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, profile, regime, family, geometry, cell = args
    _c, v2_contract, priors = _context(contract_path, elicitation_dir)
    v2.apply_learner_settings(v2_contract)
    device = v2.device_for(family)
    started = time.perf_counter()
    result = run_learner_cell(v2_contract, priors, profile, regime, family, geometry, cell, device)
    return (f"learner/{regime}/{family}/{geometry}/{cell}", result,
            {"device": device, "task_seconds": round(time.perf_counter() - started, 1)})


def _fixed_task(args: tuple[str, str, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, elicitation_dir, profile, regime, geometry, cell = args
    _c, v2_contract, priors = _context(contract_path, elicitation_dir)
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
    report = build_priors(llm_contract, log_odds)["report"]
    design = v2_contract["design"]
    cells_per_group = args.pilot_cells or int(v2_contract["profiles"][args.profile]["cells"])
    base = (str(args.contract), str(args.elicitation_dir), args.profile)
    learner_specs = [(r, f, g, cell) for r in design["regimes"] for f in design["families"][r]
                     for g in design["study_geometries"] for cell in range(cells_per_group)]
    learner_specs.sort(key=lambda s: "mlp" not in s[1])  # heaviest first, for load balance
    fixed_specs = [(r, g, cell) for r in design["regimes"] for g in design["study_geometries"]
                   for cell in range(cells_per_group)]
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
        "profile": args.profile, "cells_per_group": cells_per_group, "primary_measure": UTILITY,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        "elicitation_sha256": _sha256(args.elicitation_dir / "elicitation.jsonl"), "priors": report,
    }
    if all(p is not None for p in inputs.values()):
        o25_record = json.loads((ROOT / contract["inputs"]["utility"]["record"]).read_text(encoding="utf-8"))
        summary.update(summarize(contract, v2_contract, o25_record, load_recorded(contract, v2_contract, inputs),
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
