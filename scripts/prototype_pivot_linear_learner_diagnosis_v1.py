#!/usr/bin/env python3
"""Run the precommitted Pivot linear-learner diagnosis v1 (O35).

The pooled linear learner forecasts worse than the MLP in the linear world,
where a linear policy is correctly specified. This study records the evidence
each cross-fit fold learner actually trained on, then refits every registered
learner and linear-fit variant offline on that same evidence. It separates the
learner from the evidence it saw, and the policy class from its optimization
budget. No LLM is loaded and the hand prior fills the prior slot.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_linear_learner_diagnosis_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_prior_entry_matched_v1 as matched  # noqa: E402


pilot1 = matched.pilot1
abm1 = matched.abm1
abm2 = matched.abm2
delayed = matched.delayed
study2 = matched.study2
v3 = abm1.v3
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = matched.FIT_WORLDS
SOURCE_ARMS = ("linear_batch", "mlp_batch")
FITS = (
    "registered_linear",
    "registered_mlp",
    "linear_epochs_x10",
    "linear_learning_rate_x10",
    "linear_converged",
    "linear_converged_weak_penalty",
)
ORIGINAL_LEARNER = abm2.ProbedLearner
OPENED_SEEDS = matched.OPENED_SEEDS | {20261201, 20261202, 20261203, 20261204}


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("linear-learner diagnosis contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_linear_learner_diagnosis.v1":
        raise ValueError("unexpected linear-learner diagnosis schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("linear-learner diagnosis contract is not a precommit design")
    if tuple(contract["source_arms"]) != SOURCE_ARMS:
        raise ValueError("source arms changed")
    if tuple(row["id"] for row in contract["refits"]) != FITS:
        raise ValueError("refits changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    development = contract["profiles"]["development"]
    seeds = [int(development["base_seed"]), int(contract["probe_cohort"]["base_seed"])]
    if len(set(seeds)) != 2 or set(seeds) & OPENED_SEEDS:
        raise ValueError("the diagnosis needs fresh seed namespaces")
    if int(contract["profiles"]["smoke"]["base_seed"]) != int(development["base_seed"]):
        raise ValueError("smoke and development share the study seed namespace")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    if contract["prior"] != "HAND_RULE_IN_THE_PRIOR_SLOT":
        raise ValueError("the diagnosis runs under the hand rule")
    boundaries = contract["boundaries"]
    if boundaries.get("paid_api_calls") != 0 or boundaries.get("new_llm_calls") != 0:
        raise ValueError("the diagnosis makes no LLM calls")


# -- recording the evidence ------------------------------------------------------------


class RecordingLearner(abm2.ProbedLearner):
    """The v2 probed learner, unchanged, which lists itself when built."""

    created: list[Any] = []

    def __post_init__(self) -> None:
        super().__post_init__()
        type(self).created.append(self)


def recording_learner() -> type:
    return type("TaskRecordingLearner", (RecordingLearner,), {"created": []})


@contextlib.contextmanager
def learner_slot(learner_class: type) -> Iterator[None]:
    abm2.ProbedLearner = learner_class
    try:
        yield
    finally:
        abm2.ProbedLearner = ORIGINAL_LEARNER


def last_refit_seed(learner: Any) -> int:
    return int(
        np.random.SeedSequence((*learner.seed_components, learner.refits - 1)).generate_state(1)[0]
    )


# -- refits ----------------------------------------------------------------------------


def _settings(contract: Mapping[str, Any], fit: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    arm = fit["learner"]
    settings = dict(contract["batch_learners"][arm])
    settings.update(fit.get("overrides", {}))
    return arm, settings


def converged_linear(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    weight_decay: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Minimize the registered loss of the residual linear policy to convergence.

    Adam's weight decay adds weight_decay * theta to the gradient, so the
    registered objective is mean binary cross-entropy plus weight_decay / 2 times
    the squared norm of the residual parameters. It is strictly convex.
    """

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    policy = v3.ResidualLinear().to(dtype=torch.float64)
    inputs = torch.as_tensor(train_features, dtype=torch.float64)
    targets = torch.as_tensor(train_labels, dtype=torch.float64)
    parameters = list(policy.parameters())
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=int(max_iterations),
        tolerance_grad=1e-12,
        tolerance_change=1e-16,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def objective() -> torch.Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(policy(inputs), targets)
        return loss + 0.5 * weight_decay * sum((p**2).sum() for p in parameters)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    objective().backward()
    gradient = max(float(p.grad.abs().max()) for p in parameters)
    with torch.no_grad():
        evaluation = torch.as_tensor(eval_features, dtype=torch.float64)
        prediction = torch.sigmoid(policy(evaluation)).numpy()
        norm = float(torch.cat([p.reshape(-1) for p in parameters]).norm())
    return prediction, {"max_abs_gradient": gradient, "residual_norm": norm}


def refit(
    contract: Mapping[str, Any],
    fit: Mapping[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if fit["method"] == "REGISTERED_ADAM":
        arm, settings = _settings(contract, fit)
        prediction = abm1.fit_batch(arm, settings, train_features, train_labels, eval_features, seed)
        return prediction, {}
    if fit["method"] == "LBFGS_TO_CONVERGENCE":
        return converged_linear(
            train_features,
            train_labels,
            eval_features,
            weight_decay=float(fit["weight_decay"]),
            max_iterations=int(contract["convergence"]["max_iterations"]),
        )
    raise ValueError(f"unknown refit method: {fit['method']}")


def _scores(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    clipped = np.clip(prediction, 1e-15, 1.0 - 1e-15)
    return {
        "brier": float(np.mean((prediction - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))),
        "auc": float(roc_auc_score(labels, prediction)) if 0.0 < float(labels.mean()) < 1.0 else None,
        "sharpness": float(np.mean(np.abs(prediction - 0.5))),
    }


def run_task(args: tuple[Any, ...]) -> dict[str, Any]:
    contract, profile, replicate, world_id, source_arm = args
    config = abm1.study_config(contract, profile)
    probe_features, probe_labels = abm2.probe_cohort(contract, replicate, world_id)
    learner_class = recording_learner()
    with learner_slot(learner_class):
        run = abm2.run_arm(
            contract,
            config,
            replicate=replicate,
            visibility_id="immediate",
            world_id=world_id,
            arm_id=source_arm,
            evidence_regime="field_public",
            probe=(probe_features, probe_labels),
        )
    if abm2.ProbedLearner is not ORIGINAL_LEARNER:
        raise RuntimeError("the learner slot was not restored")
    if study2._initial_policy is not pilot1.HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot is not the hand rule")
    learners = learner_class.created
    if len(learners) != int(contract["evidence_regimes"][1]["cross_fit_folds"]):
        raise RuntimeError("expected one recorded learner per cross-fit fold")
    recorded_briers = [float(np.mean((learner.probe_output - probe_labels) ** 2)) for learner in learners]
    if float(np.mean(recorded_briers)) != run["probe_brier"]:
        raise RuntimeError("the recorded fold learners do not reproduce the run's probe Brier")
    fits_by_id = {row["id"]: row for row in contract["refits"]}
    folds = []
    for fold, learner in enumerate(learners):
        if learner.refits == 0:
            raise RuntimeError(f"fold {fold} learner saw no evidence")
        features = np.stack(learner.features)
        labels = np.asarray(learner.labels, dtype=np.float64)
        seed = last_refit_seed(learner)
        count = learner.eval_features.shape[0]
        # Parity: the source arm's registered fit on the final buffer is the recorded output.
        own = "registered_linear" if source_arm == "linear_batch" else "registered_mlp"
        parity, _ = refit(
            contract,
            fits_by_id[own],
            features,
            labels,
            np.vstack([learner.eval_features, learner.probe_features]),
            seed,
        )
        if not np.array_equal(parity[count:], learner.probe_output):
            raise RuntimeError(f"refit parity failed for {source_arm} fold {fold}")
        results = {}
        for fit_id in FITS:
            prediction, diagnostics = refit(contract, fits_by_id[fit_id], features, labels, probe_features, seed)
            results[fit_id] = {**_scores(prediction, probe_labels), **diagnostics}
        folds.append(
            {
                "fold": fold,
                "evidence_events": int(labels.size),
                "evidence_productive_share": float(labels.mean()),
                "refits": int(learner.refits),
                "fits": results,
            }
        )
    tolerance = float(contract["convergence"]["max_abs_gradient"])
    for fold in folds:
        # Only the fit at the registered penalty must converge; the weak penalty is descriptive.
        if fold["fits"]["linear_converged"]["max_abs_gradient"] > tolerance:
            raise RuntimeError(f"linear_converged did not converge in fold {fold['fold']}")
    return {
        "replicate": replicate,
        "world_id": world_id,
        "source_arm": source_arm,
        "probe_base_rate": float(probe_labels.mean()),
        "run_probe_brier": run["probe_brier"],
        "run_probe_auc": run["probe_auc"],
        "run_cumulative_entry_rate": run["cumulative_entry_rate"],
        "folds": folds,
    }


def run_study(contract: Mapping[str, Any], profile: str, workers: int = 1) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, world_id, source_arm)
        for replicate in range(config.replicates)
        for world_id in FIT_WORLDS
        for source_arm in SOURCE_ARMS
    ]
    with matched.TaskMap(workers) as task_map:
        return task_map(run_task, tasks)


# -- decisions -------------------------------------------------------------------------


def fold_mean(row: Mapping[str, Any], fit_id: str, metric: str) -> float:
    return float(np.mean([fold["fits"][fit_id][metric] for fold in row["folds"]]))


def _paired(rows, world, first, second, metric="brier") -> np.ndarray:  # noqa: ANN001
    """Per replicate, first minus second; each is (source arm, fit id)."""

    index = {(row["replicate"], row["world_id"], row["source_arm"]): row for row in rows}
    replicates = sorted({row["replicate"] for row in rows})
    return np.asarray(
        [
            fold_mean(index[(r, world, first[0])], first[1], metric)
            - fold_mean(index[(r, world, second[0])], second[1], metric)
            for r in replicates
        ],
        dtype=np.float64,
    )


def _material(values: np.ndarray, minimum: float) -> dict[str, Any]:
    bounds = matched.interval(values)
    excludes_zero = bounds["low"] > 0.0 or bounds["high"] < 0.0
    return {**bounds, "material": bool(excludes_zero and abs(bounds["mean"]) >= minimum)}


def decide_world(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], world: str) -> dict[str, Any]:
    rules = contract["decision_contract"]
    minimum = float(rules["material_brier_difference"])
    linear_e, mlp_e = "linear_batch", "mlp_batch"
    learner = {
        f"on_{source}_evidence": _material(
            _paired(rows, world, (source, "registered_linear"), (source, "registered_mlp")), minimum
        )
        for source in SOURCE_ARMS
    }
    evidence = {
        f"{fit}": _material(_paired(rows, world, (linear_e, fit), (mlp_e, fit)), minimum)
        for fit in ("registered_linear", "registered_mlp")
    }
    learner_material = all(row["material"] and row["mean"] > 0 for row in learner.values())
    evidence_material = any(row["material"] for row in evidence.values())
    if learner_material and not evidence_material:
        source = "LEARNER"
    elif evidence_material and not learner_material:
        source = "EVIDENCE"
    elif learner_material and evidence_material:
        source = "BOTH"
    else:
        source = "NEITHER"
    margin = float(rules["optimization_noninferiority_margin"])
    converged_gap = matched.interval(
        _paired(rows, world, (linear_e, "linear_converged"), (linear_e, "registered_mlp"))
    )
    closure = _material(
        _paired(rows, world, (linear_e, "registered_linear"), (linear_e, "linear_converged")), minimum
    )
    gap = learner["on_linear_batch_evidence"]
    if not (gap["material"] and gap["mean"] > 0):
        budget = "NO_GAP_TO_EXPLAIN"
    elif converged_gap["high"] <= margin:
        budget = "OPTIMIZATION_BUDGET"
    elif closure["material"] and closure["mean"] > 0:
        budget = "PARTLY_OPTIMIZATION"
    else:
        budget = "NOT_OPTIMIZATION"
    auc_gap = matched.interval(
        _paired(rows, world, (linear_e, "registered_mlp"), (linear_e, "registered_linear"), metric="auc")
    )
    return {
        "source_of_the_gap": source,
        "learner_gap": learner,
        "evidence_gap": evidence,
        "optimization": budget,
        "converged_linear_minus_mlp": converged_gap,
        "registered_minus_converged_linear": closure,
        "mlp_minus_linear_auc_on_linear_evidence": auc_gap,
    }


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    by_world = {world: decide_world(rows, contract, world) for world in FIT_WORLDS}
    primary = by_world[contract["decision_contract"]["primary_world"]]
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "o35": f"O35_{primary['source_of_the_gap']}_{primary['optimization']}",
        "primary_world": contract["decision_contract"]["primary_world"],
        "by_world": by_world,
    }


def summarize(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["source_arm"])].append(row)
    cells = []
    for world in FIT_WORLDS:
        for source in SOURCE_ARMS:
            group = grouped[(world, source)]
            fits = {
                fit_id: {
                    metric: round(float(np.mean([fold_mean(row, fit_id, metric) for row in group])), 6)
                    for metric in ("brier", "log_loss", "auc", "sharpness")
                }
                for fit_id in FITS
            }
            for fit_id in ("linear_converged", "linear_converged_weak_penalty"):
                fits[fit_id]["residual_norm"] = round(
                    float(np.mean([fold_mean(row, fit_id, "residual_norm") for row in group])), 4
                )
            cells.append(
                {
                    "world_id": world,
                    "source_arm": source,
                    "replicates": len(group),
                    "run_probe_brier": round(float(np.mean([row["run_probe_brier"] for row in group])), 6),
                    "evidence_events": round(
                        float(np.mean([f["evidence_events"] for row in group for f in row["folds"]])), 2
                    ),
                    "evidence_productive_share": round(
                        float(np.mean([f["evidence_productive_share"] for row in group for f in row["folds"]])), 4
                    ),
                    "probe_base_rate": round(float(np.mean([row["probe_base_rate"] for row in group])), 4),
                    "fits": fits,
                }
            )
    config = abm1.study_config(contract, profile)
    decisions = (
        {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        if config.replicates < int(contract["decision_contract"]["minimum_replicates"])
        else decide(rows, contract)
    )
    return {
        "schema_version": "pivot_linear_learner_diagnosis_scorecard.v1",
        "status": "SYNTHETIC_OFFLINE_REFIT_DIAGNOSIS",
        "profile": profile,
        "checks": "PASS_REFIT_PARITY_PROBE_BRIER_PARITY_AND_CONVERGENCE",
        "cells": cells,
        "tasks": len(rows),
        "decisions": decisions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot linear-learner diagnosis scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| World | Evidence from | Fit | Brier | Log loss | AUC | Sharpness |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        for fit_id, m in cell["fits"].items():
            auc = "-" if m["auc"] is None or math.isnan(m["auc"]) else f"{m['auc']:.4f}"
            lines.append(
                f"| {cell['world_id']} | {cell['source_arm']} | {fit_id} | {m['brier']:.4f} | "
                f"{m['log_loss']:.4f} | {auc} | {m['sharpness']:.4f} |"
            )
    decisions = summary["decisions"]
    lines.extend(["", f"Decision status: `{decisions['status']}`", ""])
    if decisions["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE":
        lines.append(f"- O35: `{decisions['o35']}` (primary world {decisions['primary_world']})")
        for world, row in decisions["by_world"].items():
            lines.append(f"- {world}: source `{row['source_of_the_gap']}`, optimization `{row['optimization']}`")
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
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
        "schema_version": "pivot_linear_learner_diagnosis_receipt.v1",
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
        "source_arms": list(SOURCE_ARMS),
        "refits": list(FITS),
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
    rows = run_study(contract, args.profile, workers=args.workers)
    summary = {**summarize(rows, contract, args.profile), "precommit_sha": args.precommit_sha}
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
