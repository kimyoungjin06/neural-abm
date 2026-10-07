#!/usr/bin/env python3
"""The Pivot empirical outcome-selection-trend preregistration v2 (N10).

v1's stage T showed that three yearly entry levels catch only 12 of 33 active
synthetic cells. v2 measures entry time in publication-date quarters, aligns
the persistence window to the entry quarter, and excludes entrants whose entry
date is January 1, which OpenAlex uses for year-only dates. Stage T is rerun
on twelve levels. The v1 transfer, estimator, and decision functions are
reused unchanged, verified by hash. No empirical row is read by this module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_outcome_selection_trend_v2.yaml"
V1_SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_outcome_selection_trend_v1.py"
V1_SCRIPT_SHA256 = "1a3c5befe0f34c5c488f48894867a23ce42564c32b68c60a22d423850a63a863"
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
QUARTERS_PER_YEAR = 4
WINDOW_QUARTERS = 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v1() -> ModuleType:
    if _sha256(V1_SCRIPT) != V1_SCRIPT_SHA256:
        raise RuntimeError("the v1 N10 script drifted from its bound hash")
    spec = importlib.util.spec_from_file_location("n10_v1", V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_v1()


# -- contract ----------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("the N10 v2 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_empirical_outcome_selection_trend.v2":
        raise ValueError("unexpected N10 v2 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("the N10 v2 contract is not a precommit design")
    amends = contract["amends"]
    for key in ("contract_v1", "result_v1", "script_v1"):
        if _sha256(ROOT / amends[key]["path"]) != amends[key]["sha256"]:
            raise ValueError(f"the amended {key} drifted from its bound hash")
    original = yaml.safe_load((ROOT / amends["contract_v1"]["path"]).read_text(encoding="utf-8"))
    for key in ("fields", "years", "work_eligibility", "episodes", "partition", "author_overlap", "precision", "uncertainty", "decision", "stages", "claims"):
        if contract[key] != original[key]:
            raise ValueError(f"v2 may not change the v1 {key} block")
    if contract["source"]["snapshot"] != original["source"]["snapshot"]:
        raise ValueError("v2 may not change the source snapshot")
    v1_view = {**original, "years": contract["years"], "fields": contract["fields"]}
    v1.validate_contract(v1_view)
    transfer = contract["threshold_transfer"]
    levels = int(transfer["empirical_entry_levels"])
    if levels != QUARTERS_PER_YEAR * len(contract["years"]["response_cohort_offsets"]) or contract["entry"]["entry_offsets"] != list(range(levels)):
        raise ValueError("the twelve-level clock must cover the three response years in quarters")
    if "publication_date" not in contract["source"]["required_columns"]:
        raise ValueError("the quarter clock needs publication_date")
    if contract["persistence"]["primary"]["window_quarters"] != [1, WINDOW_QUARTERS]:
        raise ValueError("the persistence window must be the twelve quarters after the entry quarter")
    for key in ("fine_clock_threshold", "feature_arms", "control_arm", "studies", "inactive_statuses", "active_status"):
        if transfer[key] != original["threshold_transfer"][key]:
            raise ValueError(f"v2 may not change the transfer {key}")


# -- the empirical clock -----------------------------------------------------------------


def quarter_index(date: dt.date) -> int:
    return QUARTERS_PER_YEAR * date.year + (date.month - 1) // 3


def is_imprecise(date: dt.date) -> bool:
    """OpenAlex records a year-only publication date as January 1."""

    return date.month == 1 and date.day == 1


def entry_clock(entry_dates: Sequence[dt.date], decision_year: int) -> dict[str, Any]:
    """The entry offset in quarters from the earliest eligible topic work of the entry year."""

    entry = min(entry_dates)
    offset = quarter_index(entry) - QUARTERS_PER_YEAR * decision_year
    if not 0 <= offset < QUARTERS_PER_YEAR * 3:
        raise ValueError(f"entry date {entry} is outside the three response years of {decision_year}")
    return {"entry_date": entry, "offset": offset, "response_year": offset // QUARTERS_PER_YEAR, "precise": not is_imprecise(entry)}


def persisted(entry_date: dt.date, follow_up_dates: Sequence[dt.date], *, minimum_works: int = 1, first: int = 1) -> bool:
    """At least `minimum_works` distinct eligible topic works in quarters first..12 after the entry quarter."""

    start = quarter_index(entry_date)
    count = sum(1 for date in follow_up_dates if start + first <= quarter_index(date) <= start + WINDOW_QUARTERS)
    return count >= minimum_works


def episode_statistic(entrants: Sequence[Mapping[str, Any]], validity: Mapping[str, Any]) -> dict[str, Any]:
    """One episode's primary statistic over precise entrants: (offset, persisted) with the v2 floors."""

    precise = [e for e in entrants if e["precise"]]
    offsets = [int(e["offset"]) for e in precise]
    values = [float(e["persisted"]) for e in precise]
    years = {offset // QUARTERS_PER_YEAR for offset in offsets}
    meets_floor = len(precise) >= int(validity["minimum_precise_eligible_entrants"]) and len(years) >= int(
        validity["minimum_nonempty_response_years"]
    )
    value = v1.spearman(offsets, values)
    return {
        "entrants": len(entrants),
        "precise_entrants": len(precise),
        "imprecise_share": (1 - len(precise) / len(entrants)) if entrants else None,
        "meets_floor": meets_floor,
        "value": value,
        "valid": meets_floor and value is not None,
    }


# -- stage T -----------------------------------------------------------------------------


def run_transfer(contract: Mapping[str, Any]) -> dict[str, Any]:
    transfer = contract["threshold_transfer"]
    studies = [v1.transfer_study(spec, transfer) for spec in transfer["studies"]]
    return {
        "schema_version": "pivot_empirical_outcome_selection_trend_transfer.v2",
        "stage": "T",
        "empirical_entry_levels": int(transfer["empirical_entry_levels"]),
        "result": v1.transfer_disposition(studies, transfer),
        "v1_three_level_data_threshold": float(transfer["v1_three_level_data_threshold"]),
        "studies": studies,
        "boundaries": dict(contract["boundaries"]),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    text = v1._markdown(payload)
    return text.replace("trend v1: stage T", "trend v2: stage T (twelve levels)")


def write_outputs(output_dir: Path, payload: Mapping[str, Any], precommit: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {"transfer.json": json.dumps(payload, indent=2, sort_keys=True) + "\n", "transfer.md": _markdown(payload)}
    for name, content in files.items():
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
        (output_dir / name).chmod(0o600)
    receipt = {
        "schema_version": "pivot_empirical_outcome_selection_trend_receipt.v2",
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
        print(json.dumps({
            "status": "PRECOMMIT_PLAN_VALID",
            "contract_sha256": _sha256(config_path),
            "empirical_entry_levels": contract["threshold_transfer"]["empirical_entry_levels"],
            "source_bindings": v1.verify_source_bindings(contract),
            "boundaries": contract["boundaries"],
        }, indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error("transfer requires a full 40-character lowercase --precommit-sha")
    if args.output_dir is None:
        parser.error("transfer requires --output-dir")
    precommit = v1.verify_precommit(args.precommit_sha, contract)
    payload = {**run_transfer(contract), "precommit_sha": args.precommit_sha}
    receipt = write_outputs(args.output_dir.resolve(), payload, precommit, config_path)
    print(json.dumps({"status": receipt["status"], "disposition": receipt["disposition"]}))


if __name__ == "__main__":
    main()
