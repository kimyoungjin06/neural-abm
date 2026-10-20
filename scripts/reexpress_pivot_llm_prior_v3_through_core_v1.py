"""N14 parity: re-express the Pivot LLM-prior pilot v3 through the policy-ladder core.

Specification: docs/research/n14-pivot-parity-v1.md, fixed before the run.

The recorded v3 study (precommit 87c90eb) ran on this workstation's CPU. Its
runs.jsonl (1,152 rows, sha256 1f532d98...) is the parity target. This runner
keeps Pivot's world: the population, network, social step, persistence
transition, probe cohort, and the fixed-rule arm. For the three field-public
learner arms, the policy-ladder semantics come from
scripts/policy_ladder_core_v1.py instead:

- the learners' prior: a core.LinearPrior built from the condition;
- the learners: core.fit_residual (linear, mlp) and
  core.fit_prior_boosted_trees, refitted with core.keyed_seed;
- the outcome record: core.OutcomeRecord, one per fold learner;
- the exploration floor: core.ExplorationFloor on core.keyed_uniform draws.

The fixed-rule arm stays in Pivot's code: its rule is computed in torch, and
the core's numpy prior differs from it in the last bits.

Pivot holds a prior in two places: as the learners' anchor, and as each
agent's own policy parameters, which the fixed rule uses and every arm
reports as mean_attention_weight. The core holds only the first. So Pivot's
prior slot stays entered for every arm, to build the agents' own parameters;
the field learners never read it. A smoke check before the freeze ran the
field arms without the slot: every decision, trajectory, and probe score was
identical, and only mean_attention_weight differed.

Modes:
- `baseline` reruns Pivot's own v3 code, to show this machine still makes
  the recorded rows;
- `core` runs the re-expression.
Both compare each row with the recorded one; a full run also compares the
whole file's sha256.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_pivot_llm_prior_pilot_v3 as pilot3  # noqa: E402


pilot1 = pilot3.pilot1
abm1 = pilot3.abm1
abm2 = pilot3.abm2
FAMILY = {"linear_batch": "linear", "mlp_batch": "mlp", "tree_batch": "tree"}
RECORDED_SHA256 = "1f532d98663767cd2c9bdfa74022b620efbac4fd4827f1d6e85a2f9bb3b3d6ca"
RECORDED_ROWS = Path("/ssd/pivot_llm_prior_pilot_v3_87c90eb_reproduction/runs.jsonl")


# -- the core's semantics in Pivot's slots ---------------------------------------------


def core_prior(condition: Mapping[str, Any], llm_prior: Mapping[str, Any]) -> core.LinearPrior:
    """The condition's prior as a core.LinearPrior.

    The hand rule is Pivot's own rule, read from its initial policy.
    """

    prior = pilot3.prior_for(condition, llm_prior)
    if prior is None:
        weight, bias = pilot1.HAND_INITIAL_POLICY()
        return core.LinearPrior(weight.detach().numpy().copy(), float(bias.detach()))
    return core.LinearPrior(np.asarray([float(v) for v in prior["weights"]], dtype=np.float64), float(prior["bias"]))


@dataclass
class CoreFieldLearner:
    """A fold learner with ProbedLearner's interface, built from core parts."""

    arm_id: str
    settings: Mapping[str, Any]
    seed_components: tuple[int, ...]
    eval_features: np.ndarray
    probe_features: np.ndarray
    excluded_agent_ids: frozenset[int] = frozenset()
    trained_agent_ids: list[int] = field(default_factory=list)
    refits: int = 0
    rule: core.LinearPrior = field(init=False)
    record: core.OutcomeRecord = field(init=False)
    output: np.ndarray = field(init=False)
    probe_output: np.ndarray = field(init=False)

    prior: ClassVar[core.LinearPrior | None] = None  # set per task by core_learner_class

    def __post_init__(self) -> None:
        if type(self).prior is None:
            raise RuntimeError("a core learner needs a task prior")
        self.rule = type(self).prior
        self.record = core.OutcomeRecord(width=self.eval_features.shape[1])
        self.output = self.rule.probability(self.eval_features)
        self.probe_output = self.rule.probability(self.probe_features)

    @property
    def labels(self) -> list[float]:
        """The outcomes held, which run_arm counts as evidence events."""

        return self.record.labels

    def update(self, events: Sequence[Any]) -> None:
        events = [event for event in events if event.agent_id not in self.excluded_agent_ids]
        if not events:
            return
        self.record.add(
            np.stack([event.features for event in events]),
            np.asarray([1.0 if event.productive else 0.0 for event in events], dtype=np.float64),
            np.asarray([int(event.agent_id) for event in events], dtype=np.int64),
            visible_from=self.refits,
        )
        self.trained_agent_ids.extend(int(event.agent_id) for event in events)
        seed = core.keyed_seed(*self.seed_components, self.refits)
        self.refits += 1
        features, labels, _agents = self.record.visible(self.refits)
        family = FAMILY[self.arm_id]
        if family == "tree":
            predictor = core.fit_prior_boosted_trees(self.rule, features, labels, self.settings, seed)
        else:
            predictor = core.fit_residual(family, self.rule, features, labels, self.settings, seed)
        prediction = predictor.predict(np.vstack([self.eval_features, self.probe_features]))
        count = self.eval_features.shape[0]
        self.output = prediction[:count]
        self.probe_output = prediction[count:]


def core_learner_class(prior: core.LinearPrior) -> type:
    return type("TaskCoreFieldLearner", (CoreFieldLearner,), {"prior": prior})


class CoreExplorationAdapter(pilot3.ExplorationCommitAdapter):
    """Pivot's exploratory entry, with the draw and test from the core."""

    def explores(self, agent_id: int) -> bool:
        draw = core.keyed_uniform(
            (self.base_seed, self.replicate, pilot3.EXPLORATION_COMPONENT, self.environment.step, agent_id), 1)
        return bool(core.ExplorationFloor(self.epsilon).explores(draw)[0])


def core_exploration_adapter(epsilon: float, base_seed: int, replicate: int) -> type:
    return type("TaskCoreExplorationAdapter", (CoreExplorationAdapter,),
                {"epsilon": float(epsilon), "base_seed": int(base_seed), "replicate": int(replicate), "created": []})


@contextlib.contextmanager
def learner_slot(learner_class: type) -> Iterator[None]:
    original = abm2.ProbedLearner
    abm2.ProbedLearner = learner_class
    try:
        yield
    finally:
        abm2.ProbedLearner = original


# -- one task ---------------------------------------------------------------------------


def run_task_core(contract: Mapping[str, Any], profile: str, replicate: int, world_id: str,
                  condition_id: str, llm_prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """pilot3.run_task with the field arms re-expressed through the core."""

    config = abm1.study_config(contract, profile)
    condition = pilot3._condition(contract, condition_id)
    share = float(condition["exploratory_share"])
    epsilon = pilot3.epsilon_for(share, config.steps - config.burn_in_steps)
    pivot_prior = pilot3.prior_for(condition, llm_prior)
    rule = core_prior(condition, llm_prior)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows, runs = [], {}
    for arm_id, evidence_regime in pilot3.ABM_ARMS:
        with contextlib.ExitStack() as stack:
            stack.enter_context(pilot1.prior_slot(pivot_prior))  # the agents' own parameters only
            if arm_id == "fixed_rule":
                adapter_class = pilot3.exploration_adapter(epsilon, config.base_seed, replicate)
            else:
                adapter_class = core_exploration_adapter(epsilon, config.base_seed, replicate)
                stack.enter_context(learner_slot(core_learner_class(rule)))
            stack.enter_context(pilot3.adapter_slot(adapter_class))
            run = abm2.run_arm(contract, config, replicate=replicate, visibility_id="immediate", world_id=world_id,
                               arm_id=arm_id, evidence_regime=evidence_regime, probe=probe)
        if len(adapter_class.created) != 1:
            raise RuntimeError("the exploration adapter was not used exactly once")
        runs[arm_id] = run
        rows.append({
            "replicate": replicate, "world_id": world_id, "condition_id": condition_id, "arm_id": arm_id,
            "evidence_regime": evidence_regime, "exploratory_share": share, "epsilon": epsilon,
            "exploratory_entry_rate": adapter_class.created[0].exploratory_entries / config.agent_count,
            **run,
        })
    if pilot3.study2._initial_policy is not pilot1.HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot was not restored")
    if pilot3.persistence.PersistenceCommitAdapter is not pilot3.ORIGINAL_ADAPTER:
        raise RuntimeError("the adapter slot was not restored")
    if abm2.ProbedLearner.__name__ != "ProbedLearner":
        raise RuntimeError("the learner slot was not restored")
    if condition_id == "hand" and world_id == "linear_fit":
        pilot3.check_nesting(contract, config, replicate=replicate, fixed_run=runs["fixed_rule"])
    return rows


def _task(args: tuple[str, Mapping[str, Any], str, int, str, str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    mode, *rest = args
    return (run_task_core if mode == "core" else pilot3.run_task)(*rest)


# -- comparison ---------------------------------------------------------------------------


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["replicate"], row["world_id"], row["condition_id"], row["arm_id"])


def compare(rows: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {_key(row): row for row in recorded}
    mismatched: dict[str, list[str]] = {}
    for row in rows:
        expected = by_key[_key(row)]
        if json.dumps(row, sort_keys=True) != json.dumps(expected, sort_keys=True):
            fields = sorted(k for k in set(row) | set(expected) if row.get(k) != expected.get(k))
            mismatched["/".join(str(part) for part in _key(row))] = fields
    by_arm: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = by_arm.setdefault(row["arm_id"], {"rows": 0, "identical": 0})
        entry["rows"] += 1
        entry["identical"] += "/".join(str(part) for part in _key(row)) not in mismatched
    return {"rows": len(rows), "identical_rows": len(rows) - len(mismatched), "by_arm": by_arm,
            "mismatched": dict(sorted(mismatched.items())[:50])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "core"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=None, help="the first R replicates; all by default")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    contract = pilot3.load_contract()
    llm_prior = pilot3.load_llm_prior(contract)
    config = abm1.study_config(contract, "development")
    replicates = config.replicates if args.replicates is None else args.replicates
    tasks = [(args.mode, contract, "development", replicate, world_id, condition_id, llm_prior)
             for replicate in range(replicates) for world_id in pilot3.FIT_WORLDS
             for condition_id in pilot3.CONDITION_IDS]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            results = [_task(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 initializer=pilot3._worker_init) as pool:
            results = list(pool.map(_task, tasks))
    rows = [row for task_rows in results for row in task_rows]
    path = args.output_dir / "runs.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    recorded = [json.loads(line) for line in RECORDED_ROWS.read_text(encoding="utf-8").splitlines()]
    if hashlib.sha256(RECORDED_ROWS.read_bytes()).hexdigest() != RECORDED_SHA256:
        raise RuntimeError("the recorded runs.jsonl changed")
    report = {
        "mode": args.mode, "replicates": replicates, "tasks": len(tasks), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "runs_jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "whole_file_identical": hashlib.sha256(path.read_bytes()).hexdigest() == RECORDED_SHA256,
        **compare(rows, recorded),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatched"}, indent=1))
    if report["mismatched"]:
        print("first mismatches:", json.dumps(dict(list(report["mismatched"].items())[:5])))


if __name__ == "__main__":
    main()
