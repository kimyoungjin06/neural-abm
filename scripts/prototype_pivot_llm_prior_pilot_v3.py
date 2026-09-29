#!/usr/bin/env python3
"""Run the precommitted Pivot local-LLM prior pilot v3 (exploration floor).

No LLM is loaded: v3 reuses the hash-bound v2 calibrated prior.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v3.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_prior_pilot_v1 as pilot1  # noqa: E402


abm1 = pilot1.abm1
abm2 = pilot1.abm2
delayed = pilot1.delayed
persistence = pilot1.persistence
study2 = pilot1.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = pilot1.FIT_WORLDS
ABM_ARMS = (
    ("fixed_rule", "none"),
    ("linear_batch", "field_public"),
    ("tree_batch", "field_public"),
    ("mlp_batch", "field_public"),
)
FIELD_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
CONDITION_IDS = ("hand", "llm_selective", "hybrid_10", "hybrid_20", "hybrid_40", "flat")
HYBRIDS = ("hybrid_10", "hybrid_20", "hybrid_40")
EXPLORATION_COMPONENT = 83
ORIGINAL_ADAPTER = persistence.PersistenceCommitAdapter
OPENED_SEEDS = pilot1.OPENED_SEEDS | {
    20261016, 20261017, 20261018, 20261019, 20261020, 20261021, 20261022,
}
NONINFERIORITY_MARGIN = -0.005


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("LLM prior v3 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_prior_pilot.v3":
        raise ValueError("unexpected LLM prior v3 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("LLM prior v3 contract is not a precommit design")
    if [row["id"] for row in contract["conditions"]] != list(CONDITION_IDS):
        raise ValueError("conditions changed")
    if [tuple(row) for row in contract["abm_arms"]] != list(ABM_ARMS):
        raise ValueError("ABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    config = contract["profiles"]["development"]
    open_steps = int(config["steps"]) - int(config["burn_in_steps"])
    if int(contract["exploration_floor"]["open_steps"]) != open_steps:
        raise ValueError("exploration open steps must match the development profile")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != pilot1.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if tuple(decisions["hybrids"]) != HYBRIDS or tuple(decisions["field_arms"]) != FIELD_ARMS:
        raise ValueError("decision cells changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this pilot")
    seeds = {int(config["base_seed"]), int(contract["probe_cohort"]["base_seed"])}
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("v3 needs fresh seed namespaces")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    boundaries = contract["boundaries"]
    if boundaries.get("paid_api_calls") != 0 or boundaries.get("new_llm_calls") != 0:
        raise ValueError("v3 makes no LLM calls")


def load_llm_prior(contract: Mapping[str, Any]) -> dict[str, Any]:
    spec = contract["v2_calibrated_priors"]
    path = Path(spec["path"])
    if pilot1._sha256(path) != spec["sha256"]:
        raise RuntimeError("the v2 calibrated priors changed")
    prior = json.loads(path.read_text(encoding="utf-8"))["priors"][spec["prior_used"]]
    return {"weights": [float(v) for v in prior["weights"]], "bias": float(prior["bias"])}


def epsilon_for(share: float, open_steps: int) -> float:
    return 1.0 - (1.0 - float(share)) ** (1.0 / int(open_steps))


# -- exploration adapter ----------------------------------------------------------


class ExplorationCommitAdapter(persistence.PersistenceCommitAdapter):
    """The persistence adapter plus prior-independent exploratory entry."""

    epsilon: float = 0.0
    base_seed: int = 0
    replicate: int = 0
    created: list[Any] = []

    def __post_init__(self) -> None:
        super().__post_init__()
        self.exploratory_entries = 0
        type(self).created.append(self)

    def explores(self, agent_id: int) -> bool:
        if self.epsilon <= 0.0:
            return False
        seed = np.random.SeedSequence(
            (self.base_seed, self.replicate, EXPLORATION_COMPONENT, self.environment.step, agent_id)
        )
        return float(np.random.default_rng(seed).random()) < self.epsilon

    def commit(self, mix_result: Any) -> Any:
        # Copied from the predecessor; only the exploratory branch is added.
        config = self.environment.config
        values = np.asarray(mix_result.mixed_values, dtype=np.float64)
        committed: list[int] = []
        losses = [0.0 for _agent in self.agents]
        transitions_open = self.environment.step > config.burn_in_steps
        self.exits_this_step = []
        for index, agent in enumerate(self.agents):
            if agent.base.pivoted:
                self._evaluate_failed_exit(index, agent)
                continue
            before = float(agent.base.pivot_readiness)
            after = float(np.clip(values[index], 0.0, 1.0))
            agent.base.pivot_readiness = after
            losses[index] = abs(after - before)
            committed.append(agent.agent_id)
            ready = after >= config.pivot_threshold
            exploring = transitions_open and not ready and self.explores(agent.agent_id)
            if transitions_open and (ready or exploring):
                if exploring:
                    self.exploratory_entries += 1
                agent.base.pivoted = True
                agent.base.productive_pivot = (
                    agent.base.productive_fit >= config.productive_threshold
                )
                agent.pivot_step = self.environment.step
                self.entry_steps[index] = self.environment.step
                self.states[index] = (
                    persistence.ACTIVE_PRODUCTIVE
                    if agent.base.productive_pivot
                    else persistence.ACTIVE_FAILED
                )
                self.environment.new_events.append(
                    study2.PivotEvent(
                        agent_id=agent.agent_id,
                        features=agent.features.copy(),
                        productive=agent.base.productive_pivot,
                        step=self.environment.step,
                    )
                )
        return study2.CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


def exploration_adapter(epsilon: float, base_seed: int, replicate: int) -> type:
    return type(
        "TaskExplorationAdapter",
        (ExplorationCommitAdapter,),
        {"epsilon": float(epsilon), "base_seed": int(base_seed), "replicate": int(replicate), "created": []},
    )


@contextlib.contextmanager
def adapter_slot(adapter_class: type) -> Iterator[None]:
    persistence.PersistenceCommitAdapter = adapter_class
    try:
        yield
    finally:
        persistence.PersistenceCommitAdapter = ORIGINAL_ADAPTER


# -- ABM --------------------------------------------------------------------------


def _condition(contract: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    return next(row for row in contract["conditions"] if row["id"] == condition_id)


def prior_for(condition: Mapping[str, Any], llm_prior: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if condition["prior"] == "hand":
        return None
    if condition["prior"] == "llm_base_rate":
        return llm_prior
    if condition["prior"] == "flat":
        return {"weights": [0.0] * len(pilot1.FEATURES), "bias": 0.0}
    raise ValueError(f"unknown prior: {condition['prior']}")


def check_nesting(
    contract: Mapping[str, Any], config: Any, *, replicate: int, fixed_run: Mapping[str, Any]
) -> None:
    predecessor = persistence.run_persistence_arm(
        scenario=abm1._scenario(contract),
        config=config,
        visibility=abm1._visibility(contract, "immediate"),
        transition=contract["common_environment"]["transition"],
        replicate=replicate,
        learning_mode="frozen",
    )
    for key, value in predecessor.items():
        if fixed_run[key] != value:
            raise RuntimeError(f"fixed_rule does not nest the persistence arm at {key}")


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    world_id: str,
    condition_id: str,
    llm_prior: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    condition = _condition(contract, condition_id)
    share = float(condition["exploratory_share"])
    epsilon = epsilon_for(share, config.steps - config.burn_in_steps)
    prior = prior_for(condition, llm_prior)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows, runs = [], {}
    for arm_id, evidence_regime in ABM_ARMS:
        adapter_class = exploration_adapter(epsilon, config.base_seed, replicate)
        with pilot1.prior_slot(prior), adapter_slot(adapter_class):
            run = abm2.run_arm(
                contract,
                config,
                replicate=replicate,
                visibility_id="immediate",
                world_id=world_id,
                arm_id=arm_id,
                evidence_regime=evidence_regime,
                probe=probe,
            )
        if len(adapter_class.created) != 1:
            raise RuntimeError("the exploration adapter was not used exactly once")
        runs[arm_id] = run
        rows.append(
            {
                "replicate": replicate,
                "world_id": world_id,
                "condition_id": condition_id,
                "arm_id": arm_id,
                "evidence_regime": evidence_regime,
                "exploratory_share": share,
                "epsilon": epsilon,
                "exploratory_entry_rate": adapter_class.created[0].exploratory_entries
                / config.agent_count,
                **run,
            }
        )
    if study2._initial_policy is not pilot1.HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot was not restored")
    if persistence.PersistenceCommitAdapter is not ORIGINAL_ADAPTER:
        raise RuntimeError("the adapter slot was not restored")
    if condition_id == "hand" and world_id == "linear_fit":
        check_nesting(contract, config, replicate=replicate, fixed_run=runs["fixed_rule"])
    return rows


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_study(
    contract: Mapping[str, Any], profile: str, llm_prior: Mapping[str, Any], workers: int = 1
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, world_id, condition_id, llm_prior)
        for replicate in range(config.replicates)
        for world_id in FIT_WORLDS
        for condition_id in CONDITION_IDS
    ]
    if workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            results = [_run_task_args(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_worker_init
        ) as pool:
            results = list(pool.map(_run_task_args, tasks))
    return [row for rows in results for row in rows]


# -- decisions ----------------------------------------------------------------------


def _values(indexed, replicates, world, condition, arm, metric) -> np.ndarray:  # noqa: ANN001
    return np.asarray(
        [float(indexed[(r, world, condition, arm)][metric]) for r in replicates],
        dtype=np.float64,
    )


def cell_decision(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    contract: Mapping[str, Any],
    *,
    condition: str,
    world: str,
    arm: str,
) -> dict[str, Any]:
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    share = "productive_share_of_entrants"
    precision = abm1._judge(
        _values(indexed, replicates, world, condition, arm, share)
        - _values(indexed, replicates, world, "hand", arm, share),
        float(decision["precision_versus_hand"]["improvement_at_least"]),
        required,
    )
    learning_gap = _values(indexed, replicates, world, "hand", arm, "probe_brier") - _values(
        indexed, replicates, world, condition, arm, "probe_brier"
    )
    noninferior = {
        "hand_minus_condition_probe_brier": pilot1.abm1.v2._distribution(learning_gap),
        "pass": float(np.mean(learning_gap)) >= NONINFERIORITY_MARGIN,
    }
    restored = abm1._judge(
        _values(indexed, replicates, world, "llm_selective", arm, "probe_brier")
        - _values(indexed, replicates, world, condition, arm, "probe_brier"),
        float(decision["learning_restored_versus_selective"]["improvement_at_least"]),
        required,
    )
    if precision["pass"] and noninferior["pass"]:
        disposition = "HYBRID_DOMINATES_HAND"
    elif precision["pass"]:
        disposition = "SELECTION_ONLY"
    elif noninferior["pass"]:
        disposition = "LEARNING_ONLY"
    else:
        disposition = "NEITHER"
    return {
        "disposition": disposition,
        "precision_versus_hand": precision,
        "learning_noninferior_versus_hand": noninferior,
        "learning_restored_versus_selective": restored,
    }


def _summary_label(count: int, total: int) -> str:
    if count == total:
        return "DOMINATES_IN_ALL_CELLS"
    if count * 2 > total:
        return "DOMINATES_IN_MAJORITY"
    if count > 0:
        return "DOMINATES_IN_MINORITY"
    return "DOMINATES_NOWHERE"


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), row["world_id"], row["condition_id"], row["arm_id"]): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    by_condition: dict[str, Any] = {}
    for condition in ("llm_selective", *HYBRIDS, "flat"):
        cells = {
            f"{world}/{arm}": cell_decision(
                indexed, replicates, contract, condition=condition, world=world, arm=arm
            )
            for world in FIT_WORLDS
            for arm in FIELD_ARMS
        }
        dominated = sorted(key for key, cell in cells.items() if cell["disposition"] == "HYBRID_DOMINATES_HAND")
        by_condition[condition] = {
            "role": "REGISTERED_HYBRID" if condition in HYBRIDS else "REFERENCE",
            "summary": _summary_label(len(dominated), len(cells)),
            "dominating_cells": dominated,
            "cells": cells,
        }
    return {"status": "FROZEN_SYNTHETIC_DECISION_COMPLETE", "by_condition": by_condition}


def summarize(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    metrics = (
        "cumulative_entry_rate",
        "exploratory_entry_rate",
        "productive_share_of_entrants",
        "probe_brier",
        "probe_auc",
        "active_failed_rate",
        "evidence_events_per_learner",
    )
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["condition_id"], row["arm_id"])].append(row)
    cells = []
    for world in FIT_WORLDS:
        for condition in CONDITION_IDS:
            for arm, evidence in ABM_ARMS:
                cell_rows = grouped[(world, condition, arm)]
                means = {}
                for metric in metrics:
                    values = [row[metric] for row in cell_rows if row[metric] is not None]
                    means[metric] = round(float(np.mean(values)), 6) if values else None
                cells.append(
                    {
                        "world_id": world,
                        "condition_id": condition,
                        "arm_id": arm,
                        "evidence_regime": evidence,
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
        "schema_version": "pivot_llm_prior_pilot_scorecard.v3",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "epsilon_by_condition": {
            row["id"]: epsilon_for(row["exploratory_share"], config.steps - config.burn_in_steps)
            for row in contract["conditions"]
        },
        "cells": cells,
        "abm_runs": len(rows),
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot local-LLM prior pilot scorecard v3",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| World | Condition | Arm | Entry | Exploratory | Entrant precision | Probe Brier | Evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        m = cell["means"]
        evidence = m["evidence_events_per_learner"]
        lines.append(
            f"| {cell['world_id']} | {cell['condition_id']} | {cell['arm_id']} | "
            f"{m['cumulative_entry_rate']:.4f} | {m['exploratory_entry_rate']:.4f} | "
            f"{m['productive_share_of_entrants']:.4f} | {m['probe_brier']:.4f} | "
            f"{'-' if evidence is None else f'{evidence:.1f}'} |"
        )
    dispositions = summary["dispositions"]
    if "by_condition" in dispositions:
        lines.extend(["", "## Frozen dispositions", ""])
        for condition, entry in dispositions["by_condition"].items():
            lines.append(
                f"- {condition} ({entry['role']}): `{entry['summary']}`, dominating cells "
                f"{entry['dominating_cells'] or '-'}"
            )
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
        "schema_version": "pivot_llm_prior_pilot_receipt.v3",
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
    config = abm1.study_config(contract, "development")
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": pilot1._sha256(config_path),
        "conditions": list(CONDITION_IDS),
        "epsilon_by_condition": {
            row["id"]: epsilon_for(row["exploratory_share"], config.steps - config.burn_in_steps)
            for row in contract["conditions"]
        },
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
    llm_prior = load_llm_prior(contract)
    rows = run_study(contract, args.profile, llm_prior, workers=args.workers)
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
