"""Diagnose D2: why was the adaptive arm worse than the table? (exploratory)

D2 found the adaptive arm (L0) worse than the four-donor table (T0-4) in both
shift controls. Two candidate causes were named but not separated:

(a) ensembling: T0-4 averages four donors' policies, so it carries four runs
    of evidence and the variance reduction of an ensemble;
(b) no forgetting: the learners refit on every visible outcome, so a shift is
    diluted by pre-shift evidence.

This diagnosis varies each factor on its own, in the shift controls only:

- (a) a single-donor table T0-1 against T0-4, and L0 against T0-1;
- (b) a forgetting learner L0-window, trained on outcomes visible in the last
  WINDOW steps, against the full learner L0-full.

Reported per arm: share precision overall and by item cohort (born before or
after the run's shift step), and each policy's effective credibility weight
over time, aligned on the shift. No study cell is scored. Nothing is
registered; this informs the revision of D2.
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

import power_info_spread_policy_ladder_v1 as power  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


CONTROLS = {"POS_flat": 0.0, "POS_flip": -3.0}
WINDOW = 12
DONORS = power.DONOR_SETS[0]
TAG_DIAGNOSIS = 3
PROBE_EVERY = 4
RELATIVE_BINS = (-16, -12, -8, -4, 0, 4, 8, 12, 16, 20)


def _learner(world: spread.World, family: str, cell: int, replicate: int, window: int | None) -> spread.Policy:
    config = world.config
    components = (TAG_DIAGNOSIS, cell, ord(config.regime), spread.FAMILIES_INDEX[family], replicate)
    prior = spread.hand_prior(config)
    if config.regime == "A":
        return spread.pooled_learning_policy(family, prior, seed_components=components, window=window)
    return spread.per_agent_learning_policy(family, prior, config.agents, seed_components=components, window=window)


def _cohort_precision(stream: spread.Stream, result: spread.RunResult) -> dict[str, float]:
    after = stream.born >= stream.shift_step
    out = {}
    for name, cohort in (("overall", np.ones_like(after)), ("pre_shift", ~after), ("post_shift", after)):
        shares = result.decision_shares[cohort]
        total = shares.sum()
        out[name] = float(shares[stream.veracity[cohort]].sum() / total) if total else float("nan")
    return out


def _sensitivity_series(config: spread.WorldConfig, probability_at) -> list[float]:
    return [
        spread.credibility_sensitivity(config, probability_at(step))
        for step in range(0, config.steps, PROBE_EVERY)
    ]


def _table_probability(donors: Sequence[Any], step: int):
    snapshots = [donor.active(step) for donor in donors]
    return lambda ids, rows: np.mean(
        [spread.snapshot_probability(s)(ids, rows) for s in snapshots], axis=0
    )


def run_cell(
    control: str, regime: str, family: str, cell: int, *,
    replicates: int = 16, sizes: Mapping[str, Any] = power.SIZES,
) -> dict[str, Any]:
    config = spread.WorldConfig(
        **sizes, regime=regime, geometry="linear",
        world_seed=power.CELL_SEED_OFFSET + 500 + cell,
        shift_credibility_weight=CONTROLS[control],
    )
    world = spread.build_world(config)
    donors = [
        spread.simulate(world, spread.build_stream(world, r), _learner(world, family, cell, r, None), r).snapshots
        for r in DONORS
    ]
    rows = []
    for replicate in range(replicates):
        stream = spread.build_stream(world, replicate)
        arms = {
            "L0_full": _learner(world, family, cell, replicate, None),
            "L0_window": _learner(world, family, cell, replicate, WINDOW),
            "T0_1": spread.table_policy("T0_1", donors[:1], final_only=False),
            "T0_4": spread.table_policy("T0_4", donors, final_only=False),
        }
        row: dict[str, Any] = {"replicate": replicate, "shift_step": stream.shift_step, "arms": {}}
        for name, policy in arms.items():
            result = spread.simulate(world, stream, policy, replicate)
            if name.startswith("L0"):
                series = _sensitivity_series(
                    config, lambda step, r=result: spread.snapshot_probability(r.snapshots.active(step))
                )
            else:
                table_donors = donors[:1] if name == "T0_1" else donors
                series = _sensitivity_series(config, lambda step, d=table_donors: _table_probability(d, step))
            row["arms"][name] = {
                "precision": _cohort_precision(stream, result),
                "false_reach": result.behavior["false_reach"],
                "credibility_weight": series,
                "seconds": round(result.seconds, 2),
            }
        rows.append(row)
    return {"control": control, "regime": regime, "family": family, "cell": cell, "rows": rows}


def _task(args: tuple[str, str, str, int, int]) -> dict[str, Any]:
    control, regime, family, cell, replicates = args
    return run_cell(control, regime, family, cell, replicates=replicates)


# -- summary --------------------------------------------------------------------


def _cluster_interval(values_by_cell: Sequence[float], key: Sequence[int]) -> list[float]:
    """95% bootstrap interval of the mean over cells (cells resampled)."""

    values = np.asarray(values_by_cell, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence(tuple(key)))
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(2000)]
    return [round(float(q), 4) for q in np.quantile(means, [0.025, 0.975])]


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(f'{cell["control"]}/{cell["regime"]}/{cell["family"]}', []).append(cell)
    contrasts = {
        "L0_full - T0_4": ("L0_full", "T0_4"),
        "L0_full - T0_1 (closed loop at matched evidence)": ("L0_full", "T0_1"),
        "T0_1 - T0_4 (ensembling)": ("T0_1", "T0_4"),
        "L0_window - L0_full (forgetting)": ("L0_window", "L0_full"),
    }
    out = {}
    for index, (key, members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        arms = members[0]["rows"][0]["arms"].keys()
        entry["mean_precision"] = {
            arm: {
                cohort: round(float(np.nanmean([r["arms"][arm]["precision"][cohort] for c in members for r in c["rows"]])), 4)
                for cohort in ("overall", "pre_shift", "post_shift")
            }
            for arm in arms
        }
        entry["contrasts"] = {}
        for position, (label, (left, right)) in enumerate(contrasts.items()):
            per_cohort = {}
            for cohort in ("overall", "pre_shift", "post_shift"):
                by_cell = [
                    float(np.nanmean([r["arms"][left]["precision"][cohort] - r["arms"][right]["precision"][cohort] for r in c["rows"]]))
                    for c in members
                ]
                per_cohort[cohort] = {
                    "mean": round(float(np.mean(by_cell)), 4),
                    "interval_95": _cluster_interval(by_cell, (TAG_DIAGNOSIS, index, position, len(cohort))),
                }
            entry["contrasts"][label] = per_cohort
        # Effective credibility weight, binned by steps relative to the shift.
        weights: dict[str, dict[str, list[float]]] = {arm: {} for arm in arms}
        for c in members:
            for r in c["rows"]:
                steps = np.arange(0, len(r["arms"]["L0_full"]["credibility_weight"])) * PROBE_EVERY
                relative = steps - r["shift_step"]
                for arm in arms:
                    for rel, value in zip(relative, r["arms"][arm]["credibility_weight"], strict=True):
                        index_bin = int(np.searchsorted(RELATIVE_BINS, rel, side="right")) - 1
                        if 0 <= index_bin < len(RELATIVE_BINS) - 1:
                            label = f"[{RELATIVE_BINS[index_bin]},{RELATIVE_BINS[index_bin + 1]})"
                            weights[arm].setdefault(label, []).append(value)
        entry["credibility_weight_by_steps_from_shift"] = {
            arm: {label: round(float(np.mean(v)), 3) for label, v in bins.items()}
            for arm, bins in weights.items()
        }
        out[key] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=16)
    args = parser.parse_args()
    order = sorted(
        ((c, r, f) for c in CONTROLS for r in power.REGIMES for f in power.FAMILIES),
        key=lambda t: (t[1] != "B", t[2] != "mlp"),
    )
    tasks = [(c, r, f, cell, args.replicates) for c, r, f in order for cell in range(args.cells)]
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
        "status": "EXPLORATORY_DIAGNOSIS_OF_D2_CONTROLS_ONLY",
        "window": WINDOW, "cells_per_group": args.cells, "replicates": args.replicates,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "groups": summarize(cells),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"wall_seconds": summary["wall_seconds"]}))


if __name__ == "__main__":
    main()
