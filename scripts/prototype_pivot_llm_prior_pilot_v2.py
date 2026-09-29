#!/usr/bin/env python3
"""Run the precommitted Pivot local-LLM prior pilot v2 (calibrated priors).

No LLM is loaded: v2 reuses the hash-bound v1 elicitation artifacts.
"""

from __future__ import annotations

import argparse
import json
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
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v2.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_prior_pilot_v1 as pilot1  # noqa: E402


abm1 = pilot1.abm1
delayed = pilot1.delayed
study2 = pilot1.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = pilot1.FIT_WORLDS
ABM_ARMS = pilot1.ABM_ARMS
FIELD_ARMS = pilot1.FIELD_ARMS
FEATURES = pilot1.FEATURES
PRIOR_IDS = (
    "hand",
    "llm_uncalibrated",
    "llm_base_rate",
    "llm_hand_level",
    "llm_verbalized",
    "oracle_linear",
    "flat",
)
VARIANTS = ("llm_base_rate", "llm_hand_level", "llm_verbalized", "llm_uncalibrated")
OPENED_SEEDS = pilot1.OPENED_SEEDS | {20261016, 20261017, 20261018, 20261019}
SELF_SEALING_ENTRY = 0.05


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("LLM prior v2 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_prior_pilot.v2":
        raise ValueError("unexpected LLM prior v2 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("LLM prior v2 contract is not a precommit design")
    if [row["id"] for row in contract["priors"]] != list(PRIOR_IDS):
        raise ValueError("prior set changed")
    if [tuple(row) for row in contract["abm_arms"]] != list(ABM_ARMS):
        raise ValueError("ABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    decisions = contract["decision_contract"]
    if decisions["primary_variant"] != VARIANTS[0] or tuple(
        [decisions["primary_variant"], *decisions["secondary_variants"]]
    ) != VARIANTS:
        raise ValueError("decision variants changed")
    if decisions.get("improvement_rule") != pilot1.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this pilot")
    if contract["calibration"]["information_used"] != "AGGREGATE_ONLY_NO_INDIVIDUAL_LABELS":
        raise ValueError("calibration must use aggregate information only")
    oracle = next(row for row in contract["priors"] if row["id"] == "oracle_linear")
    seeds = [
        int(oracle["oracle_cohort_base_seed"]),
        int(contract["profiles"]["development"]["base_seed"]),
        int(contract["probe_cohort"]["base_seed"]),
    ]
    if len(set(seeds)) != 3 or set(seeds) & OPENED_SEEDS:
        raise ValueError("v2 needs fresh seed namespaces")
    v2_learners = pilot1.abm2.load_contract()["batch_learners"]
    for arm_id in pilot1.abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    boundaries = contract["boundaries"]
    if boundaries.get("paid_api_calls") != 0 or boundaries.get("new_llm_calls") != 0:
        raise ValueError("v2 makes no LLM calls")


# -- priors ---------------------------------------------------------------------


def load_v1_elicitation(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    elicitation = contract["v1_elicitation"]
    directory = Path(elicitation["directory"])
    for name, expected in elicitation["artifact_sha256"].items():
        if pilot1._sha256(directory / name) != expected:
            raise RuntimeError(f"v1 elicitation artifact {name} changed")
    records = [
        json.loads(line)
        for line in (directory / "elicitation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    llm_prior = json.loads((directory / "llm_prior.json").read_text(encoding="utf-8"))
    return records, llm_prior


def mean_probability(weights: Sequence[float], bias: float, features: np.ndarray) -> float:
    logits = features @ np.asarray(weights, dtype=np.float64) + bias
    return float(np.mean(1.0 / (1.0 + np.exp(-logits))))


def calibrate_intercept(
    weights: Sequence[float],
    bias: float,
    features: np.ndarray,
    target: float,
    *,
    interval: Sequence[float],
    tolerance: float,
) -> float:
    """Shift the intercept until the mean prior probability equals the target."""

    low, high = bias + float(interval[0]), bias + float(interval[1])
    if not mean_probability(weights, low, features) < target < mean_probability(
        weights, high, features
    ):
        raise RuntimeError("the calibration target is outside the bisection interval")
    for _iteration in range(400):
        middle = 0.5 * (low + high)
        value = mean_probability(weights, middle, features)
        if abs(value - target) < tolerance:
            return middle
        if value < target:
            low = middle
        else:
            high = middle
    raise RuntimeError("intercept calibration did not converge")


def build_priors(
    contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    llm_prior: Mapping[str, Any],
) -> dict[str, Any]:
    if study2._initial_policy is not pilot1.HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot must hold the hand prior while priors are built")
    calibration = contract["calibration"]
    features = np.asarray([row["features"] for row in records], dtype=np.float64)
    weights = [float(value) for value in llm_prior["weights"]]
    bias = float(llm_prior["bias"])
    hand_level = float(np.mean(abm1._prior_output(features)))
    targets = {
        "llm_base_rate": float(calibration["stated_population_base_rate"]),
        "llm_hand_level": hand_level,
    }
    priors: dict[str, Any] = {"llm_uncalibrated": {"weights": weights, "bias": bias}}
    diagnostics: dict[str, Any] = {
        "hand_mean_over_design_cohort": hand_level,
        "uncalibrated_mean_over_design_cohort": mean_probability(weights, bias, features),
    }
    for prior_id, target in targets.items():
        shifted = calibrate_intercept(
            weights,
            bias,
            features,
            target,
            interval=calibration["bisection_interval"],
            tolerance=float(calibration["tolerance"]),
        )
        priors[prior_id] = {"weights": list(weights), "bias": shifted}
        diagnostics[prior_id] = {
            "target_mean": target,
            "achieved_mean": mean_probability(weights, shifted, features),
            "intercept_shift": shifted - bias,
        }
    verbalized_spec = next(row for row in contract["priors"] if row["id"] == "llm_verbalized")
    verbalized = np.asarray([row["verbalized_probability"] for row in records], dtype=np.float64)
    if not np.all(np.isfinite(verbalized)):
        raise RuntimeError("a verbalized readout is missing")
    fitted = pilot1.distill(
        features,
        pilot1.log_odds(verbalized),
        float(verbalized_spec["ridge_lambda_per_profile"]),
    )
    priors["llm_verbalized"] = {"weights": fitted["weights"], "bias": fitted["bias"]}
    diagnostics["llm_verbalized"] = {
        "r_squared_on_log_odds": fitted["r_squared_on_log_odds"],
        "mean_over_design_cohort": mean_probability(fitted["weights"], fitted["bias"], features),
    }
    oracles = pilot1.oracle_priors(contract)
    return {
        "priors": priors,
        "oracle_linear": oracles,
        "flat": {"weights": [0.0] * len(FEATURES), "bias": 0.0},
        "diagnostics": diagnostics,
    }


def check_priors(built: Mapping[str, Any], llm_prior: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    priors = built["priors"]
    if priors["llm_uncalibrated"] != {
        "weights": [float(v) for v in llm_prior["weights"]],
        "bias": float(llm_prior["bias"]),
    }:
        raise RuntimeError("llm_uncalibrated differs from the v1 prior")
    tolerance = float(contract["calibration"]["tolerance"])
    for prior_id in ("llm_base_rate", "llm_hand_level"):
        if priors[prior_id]["weights"] != priors["llm_uncalibrated"]["weights"]:
            raise RuntimeError(f"{prior_id} changed the v1 weights")
        diagnostic = built["diagnostics"][prior_id]
        if abs(diagnostic["achieved_mean"] - diagnostic["target_mean"]) >= tolerance:
            raise RuntimeError(f"{prior_id} missed its target mean")


def prior_for(prior_id: str, world_id: str, built: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if prior_id == "hand":
        return None
    if prior_id == "oracle_linear":
        return built["oracle_linear"][world_id]
    if prior_id == "flat":
        return built["flat"]
    return built["priors"][prior_id]


# -- ABM ------------------------------------------------------------------------


def run_study(
    contract: Mapping[str, Any], profile: str, built: Mapping[str, Any], workers: int = 1
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, world_id, prior_id, prior_for(prior_id, world_id, built))
        for replicate in range(config.replicates)
        for world_id in FIT_WORLDS
        for prior_id in PRIOR_IDS
    ]
    if workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            results = [pilot1._run_task_args(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=pilot1._worker_init
        ) as pool:
            results = list(pool.map(pilot1._run_task_args, tasks))
    return [row for rows in results for row in rows]


def _judge(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    *,
    variant: str,
    world_id: str,
    arm_id: str,
    metric: str,
    higher_is_better: bool,
    minimum: float,
    required: int,
) -> dict[str, Any]:
    ours = np.asarray([float(indexed[(r, world_id, variant, arm_id)][metric]) for r in replicates])
    hand = np.asarray([float(indexed[(r, world_id, "hand", arm_id)][metric]) for r in replicates])
    gain = ours - hand if higher_is_better else hand - ours
    return {
        "variant_over_hand": abm1._judge(gain, minimum, required),
        "hand_over_variant": abm1._judge(-gain, minimum, required),
    }


def _disposition(report: Mapping[str, Any]) -> str:
    if report["variant_over_hand"]["pass"]:
        return "VARIANT_BETTER"
    if report["hand_over_variant"]["pass"]:
        return "HAND_BETTER"
    return "NO_CLEAR_DIFFERENCE"


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), row["world_id"], row["prior_id"], row["arm_id"]): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        by_world = {}
        for world_id in FIT_WORLDS:
            common = {"variant": variant, "world_id": world_id, "required": required}
            quality = _judge(
                indexed, replicates, arm_id="fixed_rule", metric="probe_brier",
                higher_is_better=False,
                minimum=float(decision["prior_quality"]["improvement_at_least"]), **common,
            )
            local = _judge(
                indexed, replicates, arm_id="online_logistic", metric="probe_brier",
                higher_is_better=False,
                minimum=float(decision["local_rescue"]["improvement_at_least"]), **common,
            )
            pooled = {
                arm: _judge(
                    indexed, replicates, arm_id=arm, metric="probe_brier",
                    higher_is_better=False,
                    minimum=float(decision["pooled_washout"]["improvement_at_least"]), **common,
                )
                for arm in FIELD_ARMS
            }
            entry = _judge(
                indexed, replicates, arm_id="fixed_rule", metric="productive_share_of_entrants",
                higher_is_better=True,
                minimum=float(decision["entry_precision"]["improvement_at_least"]), **common,
            )
            persists = sorted(
                arm for arm, report in pooled.items() if _disposition(report) != "NO_CLEAR_DIFFERENCE"
            )
            by_world[world_id] = {
                "prior_quality": _disposition(quality),
                "local_rescue": _disposition(local),
                "pooled_washout": (
                    "PRIOR_PERSISTS_UNDER_POOLING" if persists else "PRIOR_WASHES_OUT_UNDER_POOLING"
                ),
                "prior_persists_for_arms": {arm: _disposition(pooled[arm]) for arm in persists},
                "entry_precision": _disposition(entry),
                "reports": {
                    "prior_quality": quality,
                    "local_rescue": local,
                    "pooled_washout": pooled,
                    "entry_precision": entry,
                },
            }
        by_variant[variant] = by_world
    self_sealing = {}
    for prior_id in PRIOR_IDS:
        for world_id in FIT_WORLDS:
            low_entry = sum(
                float(indexed[(r, world_id, prior_id, "fixed_rule")]["cumulative_entry_rate"])
                < SELF_SEALING_ENTRY
                for r in replicates
            )
            self_sealing[f"{prior_id}/{world_id}"] = low_entry >= required
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "primary_variant": VARIANTS[0],
        "by_variant": by_variant,
        "self_sealing": {key: value for key, value in self_sealing.items()},
        "self_sealing_priors": sorted(key for key, value in self_sealing.items() if value),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
    built: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = (
        "probe_brier",
        "probe_auc",
        "productive_share_of_entrants",
        "cumulative_entry_rate",
        "active_failed_rate",
        "evidence_events_per_learner",
    )
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["prior_id"], row["arm_id"])].append(row)
    cells = []
    for world_id in FIT_WORLDS:
        for prior_id in PRIOR_IDS:
            for arm_id, evidence_regime in ABM_ARMS:
                cell_rows = grouped[(world_id, prior_id, arm_id)]
                means = {}
                for metric in metrics:
                    values = [row[metric] for row in cell_rows if row[metric] is not None]
                    means[metric] = round(float(np.mean(values)), 6) if values else None
                cells.append(
                    {
                        "world_id": world_id,
                        "prior_id": prior_id,
                        "arm_id": arm_id,
                        "evidence_regime": evidence_regime,
                        "replicates": len(cell_rows),
                        "means": means,
                    }
                )
    config = abm1.study_config(contract, profile)
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    dispositions = (
        {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        if config.replicates < required
        else decide(rows, contract)
    )
    return {
        "schema_version": "pivot_llm_prior_pilot_scorecard.v2",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "calibration_diagnostics": built["diagnostics"],
        "cells": cells,
        "abm_runs": len(rows),
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot local-LLM prior pilot scorecard v2",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| World | Prior | Arm | Probe Brier | Entrant precision | Entry | Evidence |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        means = cell["means"]
        evidence = means["evidence_events_per_learner"]
        lines.append(
            f"| {cell['world_id']} | {cell['prior_id']} | {cell['arm_id']} | "
            f"{means['probe_brier']:.4f} | {means['productive_share_of_entrants']:.4f} | "
            f"{means['cumulative_entry_rate']:.4f} | "
            f"{'-' if evidence is None else f'{evidence:.1f}'} |"
        )
    dispositions = summary["dispositions"]
    if "by_variant" in dispositions:
        lines.extend(["", "## Frozen dispositions", ""])
        for variant, by_world in dispositions["by_variant"].items():
            for world_id, entry in by_world.items():
                lines.append(
                    f"- {variant} / {world_id}: prior `{entry['prior_quality']}`, local "
                    f"`{entry['local_rescue']}`, pooled `{entry['pooled_washout']}`, entry "
                    f"`{entry['entry_precision']}`"
                )
        lines.append(f"- Self-sealing priors: `{dispositions['self_sealing_priors']}`")
    lines.extend(
        [
            "",
            "Synthetic worlds only. This does not establish that the LLM knows how real "
            "pivots succeed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    built: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "calibrated_priors.json": json.dumps(built, indent=2, sort_keys=True) + "\n",
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in files.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_llm_prior_pilot_receipt.v2",
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
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": pilot1._sha256(config_path),
        "priors": list(PRIOR_IDS),
        "variants": list(VARIANTS),
        "abm_arms": [list(arm) for arm in ABM_ARMS],
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
    records, llm_prior = load_v1_elicitation(contract)
    built = build_priors(contract, records, llm_prior)
    check_priors(built, llm_prior, contract)
    rows = run_study(contract, args.profile, built, workers=args.workers)
    summary = {
        **summarize(rows, contract, args.profile, built),
        "precommit_sha": args.precommit_sha,
    }
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        built=built,
        precommit=precommit,
        contract_path=config_path,
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
