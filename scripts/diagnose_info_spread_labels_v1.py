"""Diagnose selective labels in the D2 controls (exploratory).

The fit-budget check left one question open. Even with the Pivot learner
settings, linear learners stay near the prior before the shift: a
credibility weight of 4.1 to 4.5 against a true 3. One candidate is selective
labels. A learner sees outcomes only for items that were shared, and the hand
rule shares credible items, so low-credibility items are rarely labelled.

This diagnosis changes only what a learner may see, in the flip control:

- S, "shares": outcomes of shared items only, as every registered arm sees;
- U, "matched": a random subset of all decided exposures, shared or
  declined, with as many rows as S. It removes the selection but keeps the
  volume;
- F, "all": outcomes of every decided exposure.

If selection is the cause, U and F both move toward the truth. If volume is
the cause, only F does. At each refit it also fits a plain logistic
regression of the label on the item features, with no prior and almost no
penalty, to read what the training rows themselves say.

The S arm uses the same seeds as L0-full in the fit-budget check, so it must
reproduce that arm. No study cell is scored, and nothing is registered.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_info_spread_d2_v1 as d2diag  # noqa: E402
import power_info_spread_policy_ladder_v1 as power  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


MODES = {"S_shares": "shares", "U_matched": "matched", "F_all": "all"}
CELLS = {"linear": 10, "mlp": 5}


def _learner(world: spread.World, family: str, cell: int, replicate: int, mode: str) -> spread.Policy:
    config = world.config
    components = (d2diag.TAG_DIAGNOSIS, cell, ord(config.regime), spread.FAMILIES_INDEX[family], replicate)
    prior = spread.hand_prior(config)
    if config.regime == "A":
        return spread.pooled_learning_policy(
            family, prior, seed_components=components, label_mode=mode, diagnose_labels=True
        )
    return spread.per_agent_learning_policy(
        family, prior, config.agents, seed_components=components, label_mode=mode,
        diagnose_labels=True,
    )


def run_cell(
    regime: str, family: str, cell: int, *, replicates: int = 16,
    sizes: Mapping[str, Any] = power.SIZES,
) -> dict[str, Any]:
    config = spread.WorldConfig(
        **sizes, regime=regime, geometry="linear",
        world_seed=power.CELL_SEED_OFFSET + 500 + cell,
        shift_credibility_weight=d2diag.CONTROLS["POS_flip"],
    )
    world = spread.build_world(config)
    rows = []
    for replicate in range(replicates):
        stream = spread.build_stream(world, replicate)
        row: dict[str, Any] = {"replicate": replicate, "shift_step": stream.shift_step, "arms": {}}
        for name, mode in MODES.items():
            policy = _learner(world, family, cell, replicate, mode)
            result = spread.simulate(world, stream, policy, replicate)
            row["arms"][name] = {
                "precision": d2diag._cohort_precision(stream, result),
                "credibility_weight": d2diag._sensitivity_series(
                    config, lambda step, r=result: spread.snapshot_probability(r.snapshots.active(step))
                ),
                "labels": policy.diagnostics,
                "seconds": round(result.seconds, 2),
            }
        rows.append(row)
    return {"control": "POS_flip", "regime": regime, "family": family, "cell": cell, "rows": rows}


def _task(args: tuple[str, str, int, int]) -> dict[str, Any]:
    regime, family, cell, replicates = args
    spread.use_learner_settings("pivot")
    return run_cell(regime, family, cell, replicates=replicates)


def _last_before_shift(labels: Sequence[Mapping[str, float]], shift_step: int) -> Mapping[str, float] | None:
    before = [entry for entry in labels if entry["step"] <= shift_step]
    return before[-1] if before else None


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(f'{cell["regime"]}/{cell["family"]}', []).append(cell)
    out = {}
    for index, (key, members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members), "arms": {}}
        for name in MODES:
            pre_weight, post_weight, data_weight, coverage, rows_used = [], [], [], [], []
            for c in members:
                for r in c["rows"]:
                    arm = r["arms"][name]
                    steps = np.arange(len(arm["credibility_weight"])) * d2diag.PROBE_EVERY
                    relative = steps - r["shift_step"]
                    pre = [w for rel, w in zip(relative, arm["credibility_weight"], strict=True) if -8 <= rel < 0]
                    post = [w for rel, w in zip(relative, arm["credibility_weight"], strict=True) if 12 <= rel < 20]
                    pre_weight += pre
                    post_weight += post
                    last = _last_before_shift(arm["labels"], r["shift_step"])
                    if last is not None and np.isfinite(last["data_credibility_weight"]):
                        data_weight.append(last["data_credibility_weight"])
                        coverage.append(last["low_credibility_share"])
                        rows_used.append(last["rows"])
            entry["arms"][name] = {
                "policy_weight_8_steps_before_shift": round(float(np.mean(pre_weight)), 3),
                "policy_weight_12_to_20_steps_after_shift": round(float(np.mean(post_weight)), 3),
                "data_weight_at_last_refit_before_shift": round(float(np.mean(data_weight)), 3),
                "low_credibility_share_of_rows_before_shift": round(float(np.mean(coverage)), 3),
                "rows_at_last_refit_before_shift": round(float(np.mean(rows_used)), 1),
                "precision": {
                    cohort: round(float(np.nanmean([r["arms"][name]["precision"][cohort] for c in members for r in c["rows"]])), 4)
                    for cohort in ("pre_shift", "post_shift")
                },
            }
        contrasts = {}
        for position, (left, right) in enumerate((("U_matched", "S_shares"), ("F_all", "S_shares"))):
            contrasts[f"{left} - {right}"] = {}
            for cohort in ("pre_shift", "post_shift"):
                by_cell = [
                    float(np.nanmean([r["arms"][left]["precision"][cohort] - r["arms"][right]["precision"][cohort] for r in c["rows"]]))
                    for c in members
                ]
                contrasts[f"{left} - {right}"][cohort] = {
                    "mean": round(float(np.mean(by_cell)), 4),
                    "interval_95": d2diag._cluster_interval(by_cell, (d2diag.TAG_DIAGNOSIS, 9, index, position, len(cohort))),
                }
        entry["precision_contrasts"] = contrasts
        out[key] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=16)
    args = parser.parse_args()
    order = sorted(
        ((r, f) for r in power.REGIMES for f in power.FAMILIES),
        key=lambda t: (t[0] != "B", t[1] != "mlp"),
    )
    tasks = [(r, f, cell, args.replicates) for r, f in order for cell in range(CELLS[f])]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.workers <= 1:
        cells = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cells = list(pool.map(_task, tasks))
    (args.output_dir / "cells.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in cells), encoding="utf-8"
    )
    summary = {
        "status": "EXPLORATORY_LABEL_SELECTION_DIAGNOSIS_CONTROLS_ONLY",
        "learner_settings": "pivot", "control": "POS_flip", "cells": CELLS,
        "replicates": args.replicates, "wall_seconds": round(time.perf_counter() - started, 1),
        "groups": summarize(cells),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"wall_seconds": summary["wall_seconds"]}))


if __name__ == "__main__":
    main()
