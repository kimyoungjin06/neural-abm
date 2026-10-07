#!/usr/bin/env python3
"""The Pivot empirical outcome-selection-trend preregistration v1 (N10).

Stage T transfers the synthetic threshold to the empirical clock. It reads
only stored synthetic runs of N05, N08, and N09, coarsens each run's entry
steps to the three yearly response cohorts of the empirical design, and
recomputes the registered rule. The empirical estimator, precision projection,
and decision rule are defined here as pure functions for the later stages.
No empirical row is read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_outcome_selection_trend_v1.yaml"
N08_CONTRACT = ROOT / "experiments" / "configs" / "pivot_observable_confirmation_v1.yaml"
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_DECISION_YEARS = ((2013, 2016), (2017, 2019), (2020, 2022))
EXCLUDED_FIELDS = {27, 17}
MODIFIED_Z_MULTIPLIER = 0.6744897501960817


# -- contract ----------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("the N10 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_empirical_outcome_selection_trend.v1":
        raise ValueError("unexpected N10 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("the N10 contract is not a precommit design")
    first, last = contract["years"]["decision_years"]
    for low, high in FORBIDDEN_DECISION_YEARS:
        if not (last < low or first > high):
            raise ValueError(f"decision years overlap the forbidden window {low}-{high}")
    latest_outcome = last + max(contract["years"]["response_cohort_offsets"]) + max(contract["persistence"]["primary"]["offsets"])
    if latest_outcome != contract["years"]["latest_outcome_year"] or latest_outcome > contract["source"]["last_complete_calendar_year"]:
        raise ValueError("the persistence window is not matured in the source")
    if set(contract["fields"]["ids"]) & EXCLUDED_FIELDS:
        raise ValueError("a registered field is excluded")
    n08 = yaml.safe_load(N08_CONTRACT.read_text(encoding="utf-8"))
    registered = n08["thresholds"]["outcome_selection_trend"]
    transfer = contract["threshold_transfer"]
    if float(transfer["fine_clock_threshold"]) != float(registered["threshold"]) or registered["active_when"] != "ABOVE":
        raise ValueError("the fine-clock threshold must be the N08 registered one, unchanged")
    if int(transfer["empirical_entry_levels"]) != len(contract["entry"]["entry_offsets"]):
        raise ValueError("the coarsened clock must have one level per empirical response cohort")


def verify_source_bindings(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    verified = []
    for binding in contract["source_bindings"]:
        actual = _sha256(ROOT / binding["path"])
        if actual != binding["sha256"]:
            raise RuntimeError(f"source binding mismatch for {binding['path']}: {actual} != {binding['sha256']}")
        verified.append({"path": binding["path"], "sha256": actual, "role": binding["role"]})
    return verified


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    if FULL_SHA_PATTERN.fullmatch(commit_sha) is None:
        raise RuntimeError("transfer requires a full 40-character lowercase Git SHA")
    head = _git("rev-parse", "HEAD")
    if head != commit_sha:
        raise RuntimeError(f"precommit SHA is not HEAD: {commit_sha} != {head}")
    paths = [str(path) for path in contract["precommit"]["precommit_paths"]]
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *paths)
    if dirty:
        raise RuntimeError(f"precommit paths are dirty:\n{dirty}")
    tracked = set(_git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    missing = sorted(set(paths) - tracked)
    if missing:
        raise RuntimeError(f"precommit paths missing from commit: {missing}")
    return {"commit_sha": commit_sha, "precommit_paths": paths, "source_bindings": verify_source_bindings(contract)}


# -- the statistic -----------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """The registered statistic: Spearman's rho with average ranks, None when undefined."""

    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return None if math.isnan(value) else value


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return float(np.mean(kept)) if kept else None


# -- stage T: threshold transfer ---------------------------------------------------------


def cohort_pairs(cohorts: Sequence[Sequence[int]]) -> tuple[list[int], list[float]]:
    """Expand stored [step, count, persisting] cohorts into one (entry step, persisted) pair per entrant."""

    steps: list[int] = []
    persisted: list[float] = []
    for step, count, persisting in cohorts:
        if not 0 <= persisting <= count:
            raise ValueError("a cohort has more persisting entrants than entrants")
        steps.extend([int(step)] * int(count))
        persisted.extend([1.0] * int(persisting) + [0.0] * int(count - persisting))
    return steps, persisted


def coarsen(step: int, first: int, last: int, levels: int) -> int:
    """Equal-width bin of a full-horizon entry step on [first, last]."""

    if not first <= step <= last:
        raise ValueError(f"entry step {step} is outside the full-horizon window [{first}, {last}]")
    return (step - first) * levels // (last - first + 1)


def study_geometry(study_contract: Mapping[str, Any], profile: str) -> dict[str, int]:
    config = study_contract["profiles"][profile]
    horizon = int(study_contract["h3_adapter"]["cohort_persistence_horizon_steps"])
    return {
        "steps": int(config["steps"]),
        "first": int(config["burn_in_steps"]) + 1,
        "last": int(config["steps"]) - horizon,
        "horizon": horizon,
    }


def run_statistics(row: Mapping[str, Any], geometry: Mapping[str, int], levels: int, tolerance: float) -> dict[str, Any]:
    steps, persisted = cohort_pairs(row["cohort_persistence"])
    fine = spearman(steps, persisted)
    stored = row["outcome_selection_trend"]
    reconstructed = (fine is None and stored is None) or (
        fine is not None and stored is not None and abs(fine - float(stored)) <= tolerance
    )
    bins = [coarsen(step, geometry["first"], geometry["last"], levels) for step in steps]
    return {"reconstructed": reconstructed, "coarse": spearman(bins, persisted)}


def transfer_study(spec: Mapping[str, Any], transfer: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = ROOT / spec["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = Path(spec["runs"])
    expected = summary["execution"]["artifact_sha256"]["runs.jsonl"]
    if _sha256(runs) != expected:
        raise RuntimeError(f"{spec['study']} runs do not match the hash recorded in its summary")
    study_contract = yaml.safe_load((ROOT / spec["contract"]).read_text(encoding="utf-8"))
    geometry = study_geometry(study_contract, transfer["profile"])
    levels = int(transfer["empirical_entry_levels"])
    tolerance = float(transfer["reconstruction_tolerance"])
    feature_arms = set(transfer["feature_arms"])
    key = spec["cell_key"]
    statuses = {(cell[key], cell["visibility_id"]): cell["status"] for cell in summary["cells"]}
    feature: dict[tuple[str, str], list[float | None]] = {cell: [] for cell in statuses}
    control: dict[tuple[str, str], list[float | None]] = {cell: [] for cell in statuses}
    runs_read = failures = 0
    with runs.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stats = run_statistics(row, geometry, levels, tolerance)
            runs_read += 1
            failures += not stats["reconstructed"]
            cell = (row["clock_id"], row["visibility_id"])
            if row["arm_id"] in feature_arms:
                feature[cell].append(stats["coarse"])
            elif row["arm_id"] == transfer["control_arm"]:
                control[cell].append(stats["coarse"])
    if runs_read != int(summary["execution"]["raw_row_count"]):
        raise RuntimeError(f"{spec['study']} run count drift")
    cells = [
        {
            "study": spec["study"],
            "cell_id": cell[0],
            "visibility_id": cell[1],
            "status": status,
            "coarse_value": _mean(feature[cell]),
            "coarse_control": _mean(control[cell]),
            "feature_runs": len(feature[cell]),
            "control_runs": len(control[cell]),
        }
        for cell, status in sorted(statuses.items())
    ]
    return {
        "study": spec["study"],
        "geometry": geometry,
        "runs_sha256": expected,
        "runs_read": runs_read,
        "reconstruction_failures": failures,
        "cells": cells,
    }


def transfer_disposition(studies: Sequence[Mapping[str, Any]], transfer: Mapping[str, Any]) -> dict[str, Any]:
    cells = [cell for study in studies for cell in study["cells"]]
    inactive = [c for c in cells if c["status"] in transfer["inactive_statuses"] and c["coarse_value"] is not None]
    active = [c for c in cells if c["status"] == transfer["active_status"] and c["coarse_value"] is not None]
    fine = float(transfer["fine_clock_threshold"])
    coarse = max(c["coarse_value"] for c in inactive) if inactive else None
    threshold = max(fine, coarse) if coarse is not None else None
    failures = sum(study["reconstruction_failures"] for study in studies)
    controls = [c["coarse_control"] for c in cells if c["coarse_control"] is not None]
    control_above = [c for c in cells if c["coarse_control"] is not None and threshold is not None and c["coarse_control"] > threshold]
    if failures:
        disposition = "TRANSFER_RECONSTRUCTION_FAILED"
    elif threshold is None:
        disposition = "THRESHOLD_NOT_TRANSFERABLE_NO_INACTIVE_CELL"
    elif control_above or len(controls) != len(cells):
        disposition = "THRESHOLD_NOT_TRANSFERABLE_CONTROL"
    else:
        disposition = "THRESHOLD_TRANSFERRED"
    return {
        "disposition": disposition,
        "fine_clock_threshold": fine,
        "coarse_threshold": coarse,
        "data_threshold": threshold if disposition == "THRESHOLD_TRANSFERRED" else None,
        "reconstruction_failures": failures,
        "cells": len(cells),
        "active_cells": len(active),
        "inactive_cells": len(inactive),
        "control_cells_above_threshold": [f"{c['study']}:{c['cell_id']}:{c['visibility_id']}" for c in control_above],
        "maximum_control": max(controls) if controls else None,
        "reported_not_decided": {
            "coarse_sensitivity": (
                sum(c["coarse_value"] > threshold for c in active) / len(active) if active and threshold is not None else None
            ),
            "coarse_separation": (
                min(c["coarse_value"] for c in active) - coarse if active and coarse is not None else None
            ),
        },
    }


def run_transfer(contract: Mapping[str, Any]) -> dict[str, Any]:
    transfer = contract["threshold_transfer"]
    studies = [transfer_study(spec, transfer) for spec in transfer["studies"]]
    return {
        "schema_version": "pivot_empirical_outcome_selection_trend_transfer.v1",
        "stage": "T",
        "empirical_entry_levels": int(transfer["empirical_entry_levels"]),
        "result": transfer_disposition(studies, transfer),
        "studies": studies,
        "boundaries": dict(contract["boundaries"]),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    result = payload["result"]
    lines = [
        "# Pivot empirical outcome-selection trend v1: stage T",
        "",
        f"Disposition: `{result['disposition']}`",
        "",
        "| | Value |",
        "|---|---|",
        f"| Fine-clock threshold | {result['fine_clock_threshold']} |",
        f"| Coarse threshold (max inactive cell) | {result['coarse_threshold']} |",
        f"| Data threshold | {result['data_threshold']} |",
        f"| Cells (active / inactive) | {result['cells']} ({result['active_cells']} / {result['inactive_cells']}) |",
        f"| Maximum fixed-rule control | {result['maximum_control']} |",
        f"| Reconstruction failures | {result['reconstruction_failures']} |",
        "",
        "Synthetic runs only. No empirical row was read.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Mapping[str, Any], precommit: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {"transfer.json": json.dumps(payload, indent=2, sort_keys=True) + "\n", "transfer.md": _markdown(payload)}
    for name, content in files.items():
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
        (output_dir / name).chmod(0o600)
    receipt = {
        "schema_version": "pivot_empirical_outcome_selection_trend_receipt.v1",
        "status": "COMPLETE",
        "stage": "T",
        "disposition": payload["result"]["disposition"],
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {name: _sha256(output_dir / name) for name in files},
        "empirical_rows_read": 0,
    }
    with (output_dir / "receipt.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (output_dir / "receipt.json").chmod(0o600)
    return receipt


# -- empirical stages: pure functions ----------------------------------------------------


def surge_flags(rows: Sequence[Mapping[str, Any]], threshold: float = 3.0) -> dict[str, Any]:
    """The registered incumbent-attention surge rule for one field x decision year.

    Each row carries a topic id and its recent and reference annual counts.
    The normalization universe is every topic with a positive reference.
    """

    universe = [r for r in rows if sum(r["reference"]) > 0]
    acceleration = {
        r["topic_id"]: math.log1p(float(np.mean(r["recent"]))) - math.log1p(float(np.mean(r["reference"])))
        for r in universe
    }
    if not acceleration:
        return {"status": "NO_UNIVERSE", "surges": [], "z": {}}
    values = np.asarray(list(acceleration.values()))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0 or not math.isfinite(mad):
        return {"status": "NO_SURGE_MAD_ZERO_OR_NONFINITE", "surges": [], "z": {}}
    z = {topic: MODIFIED_Z_MULTIPLIER * (value - median) / mad for topic, value in acceleration.items()}
    by_topic = {r["topic_id"]: r for r in universe}
    surges = sorted(
        topic
        for topic, score in z.items()
        if score >= threshold and acceleration[topic] > 0 and sum(by_topic[topic]["recent"]) > sum(by_topic[topic]["reference"])
    )
    return {"status": "OK", "surges": surges, "z": z}


def select_episodes(surge_years: Mapping[str, Sequence[int]], spacing: int = 3) -> dict[str, list[int]]:
    """Per topic, the earliest surge year, then the next one at least `spacing` years later."""

    selected: dict[str, list[int]] = {}
    for topic, years in sorted(surge_years.items()):
        kept: list[int] = []
        for year in sorted(set(int(y) for y in years)):
            if not kept or year >= kept[-1] + spacing:
                kept.append(year)
        selected[topic] = kept
    return selected


def partition_families(families: Mapping[int, Sequence[str]], seed: str) -> dict[tuple[int, str], str]:
    """Within each field, the first ceil(n/3) families by SHA-256 rank go to discovery."""

    assignment: dict[tuple[int, str], str] = {}
    for field_id, subfields in sorted(families.items()):
        ranked = sorted(set(subfields), key=lambda s: hashlib.sha256(f"{seed}|{field_id}|{s}".encode()).hexdigest())
        cut = math.ceil(len(ranked) / 3)
        for index, subfield in enumerate(ranked):
            assignment[(field_id, subfield)] = "discovery" if index < cut else "confirmatory"
    return assignment


def episode_statistic(entrants: Sequence[tuple[int, int]], validity: Mapping[str, Any]) -> dict[str, Any]:
    """One episode's statistic from (entry offset, persisted) pairs, with the registered floors."""

    offsets = [int(k) for k, _p in entrants]
    persisted = [float(p) for _k, p in entrants]
    meets_floor = len(entrants) >= int(validity["minimum_eligible_entrants"]) and len(set(offsets)) >= int(
        validity["minimum_nonempty_cohorts"]
    )
    value = spearman(offsets, persisted)
    return {"entrants": len(entrants), "meets_floor": meets_floor, "value": value, "valid": meets_floor and value is not None}


def field_estimates(stats: Mapping[int, Sequence[float]]) -> tuple[dict[int, float | None], float | None]:
    fields = {field_id: _mean(values) for field_id, values in sorted(stats.items())}
    complete = all(value is not None for value in fields.values())
    return fields, (float(np.mean(list(fields.values()))) if fields and complete else None)


def bootstrap_bounds(stats: Mapping[int, Sequence[float]], resamples: int, seed: int) -> tuple[float, float]:
    """Field-stratified episode bootstrap: one-sided 95% lower and upper percentile bounds."""

    rng = np.random.default_rng(seed)
    arrays = [np.asarray(values, dtype=float) for _field, values in sorted(stats.items())]
    draws = np.empty(resamples)
    for index in range(resamples):
        draws[index] = np.mean([rng.choice(values, size=values.size, replace=True).mean() for values in arrays])
    return float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def decide(stats: Mapping[int, Sequence[float]], data_threshold: float | None, contract: Mapping[str, Any]) -> dict[str, Any]:
    """The registered primary decision on valid confirmatory episode statistics."""

    minimum = int(contract["precision"]["minimum_valid_confirmatory_episodes_per_field"])
    if set(stats) != set(contract["fields"]["ids"]) or any(len(values) < minimum for values in stats.values()):
        return {"disposition": "INVALID_FIELD_SUPPORT"}
    fields, estimate = field_estimates(stats)
    lower, upper = bootstrap_bounds(stats, int(contract["uncertainty"]["resamples"]), int(contract["uncertainty"]["seed"]))
    if data_threshold is None:
        disposition = "DESCRIPTIVE_ONLY_THRESHOLD_NOT_TRANSFERRED"
    elif estimate > data_threshold and lower > 0:
        disposition = "INDICATOR_POSITIVE"
    else:
        disposition = "NOT_POSITIVE_NO_INFERENCE"
    return {
        "disposition": disposition,
        "estimate": estimate,
        "lower_bound": lower,
        "upper_bound": upper,
        "later_cohort_decline_flag": upper < 0,
        "field_vector": fields,
    }


def precision_projection(
    discovery: Mapping[int, Sequence[float]],
    confirmatory_counts: Mapping[int, int],
    data_threshold: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the confirmatory one-sided half-width from discovery and decide go or stop."""

    rule = contract["precision"]
    if sum(len(values) for values in discovery.values()) < int(rule["minimum_valid_discovery_episodes_total"]):
        return {"disposition": "STOP_DISCOVERY_SUPPORT"}
    minimum = int(rule["minimum_valid_confirmatory_episodes_per_field"])
    if any(int(confirmatory_counts.get(field_id, 0)) < minimum for field_id in contract["fields"]["ids"]):
        return {"disposition": "STOP_INSUFFICIENT_EPISODES"}
    pooled = [value - float(np.mean(values)) for values in discovery.values() if len(values) >= 1 for value in values]
    pooled_dof = sum(len(values) - 1 for values in discovery.values() if len(values) >= 1)
    pooled_sd = math.sqrt(sum(d * d for d in pooled) / pooled_dof) if pooled_dof > 0 else float("nan")
    sds = {}
    for field_id in contract["fields"]["ids"]:
        values = discovery.get(field_id, [])
        sds[field_id] = float(np.std(values, ddof=1)) if len(values) >= 3 else pooled_sd
    se = math.sqrt(sum(sds[f] ** 2 / int(confirmatory_counts[f]) for f in sds)) / len(sds)
    half_width = 1.645 * se
    return {
        "disposition": "GO" if half_width <= data_threshold else "STOP_INSUFFICIENT_PRECISION",
        "field_sd": sds,
        "projected_se": se,
        "projected_half_width": half_width,
        "data_threshold": data_threshold,
    }


# -- CLI ---------------------------------------------------------------------------------


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": _sha256(config_path),
        "stages": [stage["id"] for stage in contract["stages"]],
        "fields": contract["fields"]["ids"],
        "decision_years": contract["years"]["decision_years"],
        "source_bindings": verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "transfer"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error("transfer requires a full 40-character lowercase --precommit-sha")
    if args.output_dir is None:
        parser.error("transfer requires --output-dir")
    precommit = verify_precommit(args.precommit_sha, contract)
    payload = {**run_transfer(contract), "precommit_sha": args.precommit_sha}
    receipt = write_outputs(args.output_dir.resolve(), payload, precommit, config_path)
    print(json.dumps({"status": receipt["status"], "disposition": receipt["disposition"]}))


if __name__ == "__main__":
    main()
