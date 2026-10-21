"""N15 parity: recorded Pivot runs through the package's policy-ladder semantics.

Specification: docs/research/n15-package-parity-v1.md, fixed before the runs.

Decision 0017 extracted the policy-ladder semantics that two consumers share
into neural_abm.policy_ladder (torch-free) and neural_abm.policy_learners
(torch). This runner places them in Pivot's learner slot and reproduces two
recorded Pivot runs on the workstation where they were recorded:

- `v3`: the LLM-prior pilot v3 (precommit 87c90eb), immediate visibility.
  - The three field-public learner arms run through the package.
  - The fixed-rule arm stays in Pivot's code, as in N14.
  - The whole runs.jsonl must equal the record.
- `abm2`: the nonlinear policy ABM v2 (precommit c56d741).
  - Only its three field-public feature arms (linear, tree, and MLP) run,
    under the immediate and two delayed-failure visibility regimes.
  - Each row must equal the recorded row.
  - With `--insertion release`, outcomes enter the package record when Pivot
    releases them, as Pivot does.
  - With `--insertion event`, they enter when they are realized, each
    tagged with the step it will be released. Pivot's release order then
    comes only from the record's ordering rule (visible step, then
    insertion), which is open item O27.

The world, the fixed-rule arm, and the agents' own parameters stay in
Pivot's code (see N14).
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

import prototype_pivot_llm_prior_pilot_v3 as pilot3  # noqa: E402
from neural_abm import policy_ladder as ladder  # noqa: E402
from neural_abm import policy_learners as learners  # noqa: E402


pilot1 = pilot3.pilot1
abm1 = pilot3.abm1
abm2 = pilot3.abm2
delayed = pilot3.delayed
FIELD_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
RECORDS = {
    "v3": (Path("/ssd/pivot_llm_prior_pilot_v3_87c90eb_reproduction/runs.jsonl"),
           "1f532d98663767cd2c9bdfa74022b620efbac4fd4827f1d6e85a2f9bb3b3d6ca"),
    "abm2": (Path("/ssd/pivot_nonlinear_policy_abm_v2_c56d741_reproduction/runs.jsonl"),
             "8088d43cbee6c5c19c1e8f08499eb734b725f5f0ccbe79c59482f00d7da4c41f"),
}


# -- the package's semantics in Pivot's slots -----------------------------------------


def package_prior(pivot_prior: Mapping[str, Any] | None) -> ladder.LinearPrior:
    """A condition's prior as a package LinearPrior; None is Pivot's hand rule."""

    if pivot_prior is None:
        weight, bias = pilot1.HAND_INITIAL_POLICY()
        return ladder.LinearPrior(weight.detach().numpy().copy(), float(bias.detach()))
    return ladder.LinearPrior(np.asarray([float(v) for v in pivot_prior["weights"]], dtype=np.float64),
                              float(pivot_prior["bias"]))


class ReleaseListener:
    """Hands each realized outcome to the field learners, with its release step."""

    learners: ClassVar[list[Any]] = []
    environment: ClassVar[Any] = None


ORIGINAL_INGEST = delayed.DelayedPivotEnvironment.ingest_private_events


def _ingest_and_announce(self: Any) -> list[Any]:
    events = ORIGINAL_INGEST(self)
    ReleaseListener.environment = self
    for event in events:
        release_step = event.step + 1 + self.visibility.delay(event)
        if any(queued is event for queued in self.release_queue.get(release_step, [])):
            for learner in ReleaseListener.learners:
                learner.realized(event, release_step)
    return events


@dataclass
class PackageFieldLearner:
    """A fold learner with ProbedLearner's interface, built from the package."""

    arm_id: str
    settings: Mapping[str, Any]
    seed_components: tuple[int, ...]
    eval_features: np.ndarray
    probe_features: np.ndarray
    excluded_agent_ids: frozenset[int] = frozenset()
    trained_agent_ids: list[int] = field(default_factory=list)
    refits: int = 0
    rule: ladder.LinearPrior = field(init=False)
    record: ladder.OutcomeRecord = field(init=False)
    output: np.ndarray = field(init=False)
    probe_output: np.ndarray = field(init=False)

    prior: ClassVar[ladder.LinearPrior | None] = None
    insertion: ClassVar[str] = "release"

    def __post_init__(self) -> None:
        if type(self).prior is None:
            raise RuntimeError("a package learner needs a task prior")
        self.rule = type(self).prior
        self.record = ladder.OutcomeRecord(width=self.eval_features.shape[1])
        self.output = self.rule.probability(self.eval_features)
        self.probe_output = self.rule.probability(self.probe_features)
        if type(self).insertion == "event":
            ReleaseListener.learners.append(self)

    def _add(self, events: Sequence[Any], visible_from: int) -> None:
        self.record.add(
            np.stack([event.features for event in events]),
            np.asarray([1.0 if event.productive else 0.0 for event in events], dtype=np.float64),
            np.asarray([int(event.agent_id) for event in events], dtype=np.int64),
            visible_from=visible_from,
        )

    def realized(self, event: Any, release_step: int) -> None:
        """Event insertion: add the outcome now, visible from its release step."""

        if event.agent_id not in self.excluded_agent_ids:
            self._add([event], release_step)

    def _now(self) -> int:
        return int(ReleaseListener.environment.step) if type(self).insertion == "event" else self.refits

    @property
    def labels(self) -> list[float]:
        """The outcomes released so far, which run_arm counts as evidence."""

        return list(self.record.visible(self._now())[1])

    def update(self, events: Sequence[Any]) -> None:
        events = [event for event in events if event.agent_id not in self.excluded_agent_ids]
        if not events:
            return
        if type(self).insertion == "release":
            self._add(events, self.refits)
        self.trained_agent_ids.extend(int(event.agent_id) for event in events)
        seed = ladder.keyed_seed(*self.seed_components, self.refits)
        visible_at = self.refits + 1 if type(self).insertion == "release" else self._now()
        self.refits += 1
        features, labels, _agents = self.record.visible(visible_at)
        stacked = np.vstack([self.eval_features, self.probe_features])
        if self.arm_id == "tree_batch":
            predictor = ladder.fit_prior_boosted_trees(self.rule, features, labels, self.settings, seed)
        else:
            predictor = learners.fit_residual_policy(
                "linear" if self.arm_id == "linear_batch" else "mlp", self.rule, features, labels,
                learning_rate=float(self.settings["learning_rate"]), epochs=int(self.settings["epochs"]),
                weight_decay=float(self.settings["weight_decay"]), hidden_width=self.settings.get("hidden_width"),
                seed=seed,
            )
        prediction = predictor.predict(stacked)
        count = self.eval_features.shape[0]
        self.output = prediction[:count]
        self.probe_output = prediction[count:]


def package_learner_class(prior: ladder.LinearPrior, insertion: str) -> type:
    return type("TaskPackageFieldLearner", (PackageFieldLearner,), {"prior": prior, "insertion": insertion})


class PackageExplorationAdapter(pilot3.ExplorationCommitAdapter):
    """Pivot's exploratory entry, with the draw and test from the package."""

    def explores(self, agent_id: int) -> bool:
        draw = ladder.keyed_uniform(
            (self.base_seed, self.replicate, pilot3.EXPLORATION_COMPONENT, self.environment.step, agent_id), 1)
        return bool(ladder.ExplorationFloor(self.epsilon).explores(draw)[0])


def package_exploration_adapter(epsilon: float, base_seed: int, replicate: int) -> type:
    return type("TaskPackageExplorationAdapter", (PackageExplorationAdapter,),
                {"epsilon": float(epsilon), "base_seed": int(base_seed), "replicate": int(replicate), "created": []})


@contextlib.contextmanager
def learner_slot(learner_class: type) -> Iterator[None]:
    original = abm2.ProbedLearner
    abm2.ProbedLearner = learner_class
    ReleaseListener.learners = []
    ReleaseListener.environment = None
    delayed.DelayedPivotEnvironment.ingest_private_events = _ingest_and_announce
    try:
        yield
    finally:
        abm2.ProbedLearner = original
        delayed.DelayedPivotEnvironment.ingest_private_events = ORIGINAL_INGEST
        ReleaseListener.learners = []
        ReleaseListener.environment = None


def _restored() -> None:
    if abm2.ProbedLearner.__name__ != "ProbedLearner":
        raise RuntimeError("the learner slot was not restored")
    if delayed.DelayedPivotEnvironment.ingest_private_events is not ORIGINAL_INGEST:
        raise RuntimeError("the environment was not restored")


# -- tasks ----------------------------------------------------------------------------


def run_v3_task(contract: Mapping[str, Any], profile: str, replicate: int, world_id: str, condition_id: str,
                llm_prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """pilot3.run_task with the field arms through the package."""

    config = abm1.study_config(contract, profile)
    condition = pilot3._condition(contract, condition_id)
    share = float(condition["exploratory_share"])
    epsilon = pilot3.epsilon_for(share, config.steps - config.burn_in_steps)
    pivot_prior = pilot3.prior_for(condition, llm_prior)
    rule = package_prior(pivot_prior)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows, runs = [], {}
    for arm_id, evidence_regime in pilot3.ABM_ARMS:
        with contextlib.ExitStack() as stack:
            stack.enter_context(pilot1.prior_slot(pivot_prior))  # the agents' own parameters only
            if arm_id == "fixed_rule":
                adapter_class = pilot3.exploration_adapter(epsilon, config.base_seed, replicate)
            else:
                adapter_class = package_exploration_adapter(epsilon, config.base_seed, replicate)
                stack.enter_context(learner_slot(package_learner_class(rule, "release")))
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
    _restored()
    if condition_id == "hand" and world_id == "linear_fit":
        pilot3.check_nesting(contract, config, replicate=replicate, fixed_run=runs["fixed_rule"])
    return rows


def run_abm2_task(contract: Mapping[str, Any], profile: str, replicate: int, visibility_id: str, world_id: str,
                  insertion: str) -> list[dict[str, Any]]:
    """The field-public feature arms of abm2.run_task, through the package."""

    config = abm2.study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows = []
    for arm_id in FIELD_ARMS:
        with learner_slot(package_learner_class(package_prior(None), insertion)):
            run = abm2.run_arm(contract, config, replicate=replicate, visibility_id=visibility_id,
                               world_id=world_id, arm_id=arm_id, evidence_regime="field_public", probe=probe)
        rows.append({"replicate": replicate, "visibility_id": visibility_id, "world_id": world_id,
                     "arm_id": arm_id, "evidence_regime": "field_public", **run})
    _restored()
    return rows


def _task(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    target, *rest = args
    return run_v3_task(*rest) if target == "v3" else run_abm2_task(*rest)


# -- comparison ---------------------------------------------------------------------------


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in ("replicate", "visibility_id", "world_id", "condition_id", "arm_id",
                                      "evidence_regime"))


def compare(rows: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {_key(row): row for row in recorded}
    mismatched: dict[str, list[str]] = {}
    by_group: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = by_key[_key(row)]
        name = "/".join(str(part) for part in _key(row) if part is not None)
        identical = json.dumps(row, sort_keys=True) == json.dumps(expected, sort_keys=True)
        if not identical:
            mismatched[name] = sorted(k for k in set(row) | set(expected) if row.get(k) != expected.get(k))
        group = by_group.setdefault(f'{row.get("visibility_id", "immediate")}/{row["arm_id"]}',
                                    {"rows": 0, "identical": 0})
        group["rows"] += 1
        group["identical"] += identical
    return {"rows": len(rows), "identical_rows": len(rows) - len(mismatched), "by_group": dict(sorted(by_group.items())),
            "mismatched": dict(sorted(mismatched.items())[:50])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("v3", "abm2"))
    parser.add_argument("--insertion", choices=("release", "event"), default="release")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=None, help="the first R replicates; all by default")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.target == "v3":
        if args.insertion != "release":
            parser.error("v3 has immediate visibility; use --insertion release")
        contract = pilot3.load_contract()
        llm_prior = pilot3.load_llm_prior(contract)
        config = abm1.study_config(contract, "development")
        replicates = config.replicates if args.replicates is None else args.replicates
        tasks = [("v3", contract, "development", r, w, c, llm_prior)
                 for r in range(replicates) for w in pilot3.FIT_WORLDS for c in pilot3.CONDITION_IDS]
    else:
        contract = abm2.load_contract()
        config = abm2.study_config(contract, "development")
        replicates = config.replicates if args.replicates is None else args.replicates
        tasks = [("abm2", contract, "development", r, v, w, args.insertion)
                 for r in range(replicates) for v in abm2.VISIBILITY_IDS for w in abm1.FIT_WORLDS]
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
    record_path, record_sha = RECORDS[args.target]
    if hashlib.sha256(record_path.read_bytes()).hexdigest() != record_sha:
        raise RuntimeError("the recorded runs.jsonl changed")
    recorded = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "target": args.target, "insertion": args.insertion, "replicates": replicates, "tasks": len(tasks),
        "workers": args.workers, "wall_seconds": round(time.perf_counter() - started, 1),
        "runs_jsonl_sha256": sha, "whole_file_identical": sha == record_sha, **compare(rows, recorded),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatched"}, indent=1))
    if report["mismatched"]:
        print("first mismatches:", json.dumps(dict(list(report["mismatched"].items())[:5])))


if __name__ == "__main__":
    main()
