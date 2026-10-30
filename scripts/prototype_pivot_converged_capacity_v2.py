#!/usr/bin/env python3
"""Run the precommitted Pivot converged capacity comparison v2 (O36).

The ladder's capacity conclusions (C07 to C14) were drawn with every residual
learner trained by 300 Adam steps from the frozen prior. O35 found that budget
leaves the linear learner far from its optimum where its class is right. This
study reruns the two studies behind those conclusions, the matched-marginal
discriminator v3 and the nonlinear policy ABM v2, changing only how each
learner is trained:
- registered: the recorded training, which must reproduce both records;
- converged: every residual objective minimized to convergence;
- tuned: each fit's training selected on its own evidence from one menu.
The two studies' own frozen decision code judges every regime. No LLM is
loaded.

v1 stopped when one inner fit missed the gradient tolerance. v2 changes only
how a fit that misses it is handled: an inner fit makes its candidate
unavailable for that selection, and an acting fit keeps its final iterate and
is counted against a registered ceiling.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import multiprocessing
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_converged_capacity_v2.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_nonlinear_policy_abm_v2 as abm2  # noqa: E402


abm1 = abm2.abm1
v3 = abm1.v3
v2 = abm1.v2
delayed = abm1.delayed
study2 = abm1.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
REGIMES = ("registered", "converged", "tuned")
RESIDUAL_ARMS = ("linear_batch", "mlp_batch")
RESIDUAL_MODELS = ("matched_linear", "residual_mlp")
ARM_KEYS = (
    ("fixed_rule", "none"),
    ("linear_batch", "neighbor_local"),
    ("tree_batch", "neighbor_local"),
    ("mlp_batch", "neighbor_local"),
    ("linear_batch", "field_public"),
    ("tree_batch", "field_public"),
    ("mlp_batch", "field_public"),
)
REGIME_ARM_KEYS = {
    "registered": ARM_KEYS,
    "converged": tuple(key for key in ARM_KEYS if key[0] in RESIDUAL_ARMS),
    "tuned": tuple(key for key in ARM_KEYS if key[0] != "fixed_rule"),
}
ADAM = "ADAM_REGISTERED_BUDGET"
LBFGS = "LBFGS_TO_CONVERGENCE"
STUDY_STATUS = {
    (True, True): "O36_HOLD_UNDER_BOTH",
    (False, True): "O36_HOLD_WHEN_TUNED_ONLY",
    (True, False): "O36_HOLD_WHEN_CONVERGED_ONLY",
    (False, False): "O36_CHANGE_UNDER_BOTH",
}
ROW_KEYS = ("replicate", "visibility_id", "world_id", "arm_id", "evidence_regime")
ORIGINAL_LEARNER = abm2.ProbedLearner
ORIGINAL_FIT_CANDIDATE = v3.fit_candidate
ORIGINAL_SELECT_CANDIDATE = v3.select_candidate
LOG_COUNTS = (
    "acting_fits",
    "inner_fits",
    "lbfgs_fits",
    "lbfgs_acting_fits",
    "lbfgs_restarts",
    "lbfgs_unconverged_acting",
    "lbfgs_unconverged_inner",
    "unavailable_candidates",
    "fallbacks",
)
ORIGINAL_INITIAL_POLICY = study2._initial_policy


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("converged capacity contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_converged_capacity.v2":
        raise ValueError("unexpected converged capacity schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("converged capacity contract is not a precommit design")
    if tuple(row["id"] for row in contract["training_regimes"]) != REGIMES:
        raise ValueError("training regimes changed")
    abm = contract["studies"]["abm"]
    if tuple(tuple(key) for key in abm["arm_keys"]) != ARM_KEYS:
        raise ValueError("ABM arm keys changed")
    if tuple(abm["residual_arms"]) != RESIDUAL_ARMS:
        raise ValueError("ABM residual arms changed")
    if tuple(contract["studies"]["policy_layer"]["residual_models"]) != RESIDUAL_MODELS:
        raise ValueError("policy-layer residual models changed")
    learners = abm2.load_contract()["batch_learners"]
    registered_decay = float(learners["linear_batch"]["weight_decay"])
    if float(learners["mlp_batch"]["weight_decay"]) != registered_decay:
        raise ValueError("the ABM residual learners no longer share one weight decay")
    converged = next(row for row in contract["training_regimes"] if row["id"] == "converged")
    if float(converged["abm_weight_decay"]) != registered_decay:
        raise ValueError("the converged regime must keep the registered ABM weight decay")
    tuning = contract["tuning"]
    menu = [(row["training"], float(row["weight_decay"])) for row in tuning["residual_menu"]]
    if (ADAM, registered_decay) not in menu:
        raise ValueError("the residual menu must nest the registered setting")
    for candidate in v3._model_contract(v3.load_contract(), "matched_linear")["candidates"]:
        if (LBFGS, float(candidate["weight_decay"])) not in menu:
            raise ValueError("the residual menu must nest every converged v3 candidate")
    tree = learners["tree_batch"]
    fixed = tuning["abm_tree_menu"]["fixed"]
    if float(fixed["leaf_l2"]) != float(tree["leaf_l2"]):
        raise ValueError("the tree menu changed the registered leaf penalty")
    registered_tree = {key: tree[key] for key in ("rounds", "learning_rate", "max_depth", "min_samples_leaf")}
    if registered_tree not in tuning["abm_tree_menu"]["candidates"]:
        raise ValueError("the tree menu must nest the registered tree")
    selection = tuning["abm_selection"]
    if int(selection["minimum_events_per_label"]) < int(selection["inner_folds"]):
        raise ValueError("every inner fold must hold an event of each label")
    for study in ("policy_layer", "abm"):
        recorded = json.loads((ROOT / contract["studies"][study]["recorded_summary"]).read_text())
        name = Path(contract["studies"][study]["recorded_rows"]).name
        if recorded["execution"]["artifact_sha256"][name] != contract["studies"][study]["recorded_rows_sha256"]:
            raise ValueError(f"{study} recorded row hash differs from its summary")
    if int(contract["decision_contract"]["minimum_replicates"]) != int(
        abm2.load_contract()["decision_contract"]["improved_replicates_at_least"]
    ):
        raise ValueError("minimum replicates differ from ABM v2")
    handling = contract["convergence"]["non_converged"]
    if handling["inner_fit"] != "ITS_CANDIDATE_IS_UNAVAILABLE_FOR_THAT_SELECTION":
        raise ValueError("an unconverged inner fit must make its candidate unavailable")
    if handling["acting_fit"] != "KEEPS_ITS_FINAL_ITERATE_AND_IS_COUNTED":
        raise ValueError("an unconverged acting fit must keep its final iterate")
    if not 0.0 < float(handling["acting_ceiling_share"]) <= 0.001:
        raise ValueError("the acting ceiling must be at most 0.1%")
    if contract["supersedes"]["status"] != "STOPPED_INVALID_IMPLEMENTATION_NO_OUTPUTS_WRITTEN":
        raise ValueError("v2 must record how v1 ended")
    boundaries = contract["boundaries"]
    if boundaries.get("paid_api_calls") != 0 or boundaries.get("new_llm_calls") != 0:
        raise ValueError("the comparison makes no LLM calls")
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside this comparison")


# -- training ----------------------------------------------------------------------------


def new_log() -> dict[str, Any]:
    return {
        **{key: 0 for key in LOG_COUNTS},
        "max_lbfgs_gradient": 0.0,
        "max_acting_gradient": 0.0,
        "selected": Counter(),
    }


def merge_logs(logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = new_log()
    for log in logs:
        for key in LOG_COUNTS:
            merged[key] += int(log[key])
        for key in ("max_lbfgs_gradient", "max_acting_gradient"):
            merged[key] = max(merged[key], float(log[key]))
        merged["selected"].update(log["selected"])
    merged["selected"] = {str(key): count for key, count in sorted(merged["selected"].items())}
    return merged


def residual_policy(kind: str, hidden_width: int | None, seed: int) -> torch.nn.Module:
    """The registered residual policy, initialized exactly as its Adam fit is."""

    if kind == "linear":
        return v3.ResidualLinear()
    if kind == "mlp":
        return v2.ResidualMLP(int(hidden_width), seed)
    raise ValueError(f"unknown residual policy: {kind}")


def converge(
    policy: torch.nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    weight_decay: float,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize the registered objective to the gradient tolerance.

    Adam's coupled weight decay adds weight_decay * theta to the gradient, so the
    objective it descends is mean binary cross-entropy plus weight_decay / 2
    times the squared norm of the trainable parameters.
    """

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    policy = policy.to(dtype=torch.float64)
    inputs = torch.as_tensor(train_features, dtype=torch.float64)
    targets = torch.as_tensor(train_labels, dtype=torch.float64)
    parameters = list(policy.parameters())
    tolerance = float(spec["max_abs_gradient"])

    def objective() -> torch.Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(policy(inputs), targets)
        return loss + 0.5 * weight_decay * sum((p**2).sum() for p in parameters)

    def gradient() -> float:
        for p in parameters:
            p.grad = None
        objective().backward()
        return max(float(p.grad.abs().max()) for p in parameters)

    restarts = 0
    for attempt in range(int(spec["restarts"]) + 1):
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=float(spec["learning_rate"]),
            max_iter=int(spec["max_iterations"]),
            tolerance_grad=tolerance,
            tolerance_change=0.0,
            history_size=int(spec["history_size"]),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        final = gradient()
        restarts = attempt
        if final <= tolerance:
            break
    with torch.no_grad():
        evaluation = torch.as_tensor(eval_features, dtype=torch.float64)
        prediction = torch.sigmoid(policy(evaluation)).numpy()
    return prediction, {"max_abs_gradient": final, "restarts": restarts, "converged": final <= tolerance}


def _record_lbfgs(log: dict[str, Any], info: Mapping[str, Any], *, acting: bool) -> None:
    gradient = float(info["max_abs_gradient"])
    log["lbfgs_fits"] += 1
    log["lbfgs_restarts"] += int(info["restarts"])
    log["max_lbfgs_gradient"] = max(log["max_lbfgs_gradient"], gradient)
    if acting:
        log["lbfgs_acting_fits"] += 1
        log["max_acting_gradient"] = max(log["max_acting_gradient"], gradient)
    if not info["converged"]:
        log["lbfgs_unconverged_acting" if acting else "lbfgs_unconverged_inner"] += 1


def abm_candidate(
    contract: Mapping[str, Any],
    arm_id: str,
    settings: Mapping[str, Any],
    candidate: Mapping[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    seed: int,
    log: dict[str, Any],
    *,
    acting: bool,
) -> tuple[np.ndarray, bool]:
    """One candidate's fit and whether it is available: only an unconverged fit is not."""

    log["acting_fits" if acting else "inner_fits"] += 1
    if arm_id == "tree_batch":
        prediction = abm1.fit_prior_boosted_trees(
            train_features, train_labels, eval_features, {**settings, **candidate}, seed
        )
        return prediction, True
    if candidate["training"] == ADAM:
        prediction = abm1.fit_batch(
            arm_id,
            {**settings, "weight_decay": float(candidate["weight_decay"])},
            train_features,
            train_labels,
            eval_features,
            seed,
        )
        return prediction, True
    if candidate["training"] != LBFGS:
        raise ValueError(f"unknown training: {candidate['training']}")
    kind = "linear" if arm_id == "linear_batch" else "mlp"
    prediction, info = converge(
        residual_policy(kind, settings.get("hidden_width"), seed),
        train_features,
        train_labels,
        eval_features,
        weight_decay=float(candidate["weight_decay"]),
        spec=contract["convergence"],
    )
    _record_lbfgs(log, info, acting=acting)
    # An unconverged acting fit keeps its final iterate; the caller drops an unconverged inner fit.
    return prediction, bool(info["converged"])


def tuned_fit(
    contract: Mapping[str, Any],
    arm_id: str,
    settings: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    agent_ids: np.ndarray,
    eval_features: np.ndarray,
    seed: int,
    log: dict[str, Any],
) -> np.ndarray:
    tuning = contract["tuning"]
    selection = tuning["abm_selection"]
    menu = (
        tuning["residual_menu"]
        if arm_id in RESIDUAL_ARMS
        else [{**tuning["abm_tree_menu"]["fixed"], **row} for row in tuning["abm_tree_menu"]["candidates"]]
    )
    folds = int(selection["inner_folds"])
    if min(int(np.sum(labels == 1.0)), int(np.sum(labels == 0.0))) < int(selection["minimum_events_per_label"]):
        # Too little evidence to cross-validate: the registered setting, unselected.
        log["fallbacks"] += 1
        log["acting_fits"] += 1
        return abm1.fit_batch(arm_id, settings, features, labels, eval_features, seed)
    assignment = v3.inner_folds(labels, agent_ids, folds)
    briers = np.zeros((len(menu), folds), dtype=np.float64)
    available = np.ones(len(menu), dtype=bool)
    for fold in range(folds):
        fold_seed = int(np.random.SeedSequence((seed, fold + 1)).generate_state(1)[0])
        fit_rows = assignment != fold
        for position, candidate in enumerate(menu):
            if not available[position]:
                continue
            prediction, converged = abm_candidate(
                contract,
                arm_id,
                settings,
                candidate,
                features[fit_rows],
                labels[fit_rows],
                features[~fit_rows],
                fold_seed,
                log,
                acting=False,
            )
            if not converged:
                # A candidate whose objective could not be minimized here is unavailable for this selection.
                available[position] = False
                log["unavailable_candidates"] += 1
                continue
            briers[position, fold] = float(np.mean((prediction - labels[~fit_rows]) ** 2))
    selected = v3.select_candidate(np.where(available, briers.mean(axis=1), np.inf))
    log["selected"][selected] += 1
    prediction, _converged = abm_candidate(
        contract, arm_id, settings, menu[selected], features, labels, eval_features, seed, log, acting=True
    )
    return prediction


def train_abm(
    contract: Mapping[str, Any],
    regime: str,
    arm_id: str,
    settings: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    agent_ids: np.ndarray,
    eval_features: np.ndarray,
    seed: int,
    log: dict[str, Any],
) -> np.ndarray:
    if regime == "registered":
        log["acting_fits"] += 1
        return abm1.fit_batch(arm_id, settings, features, labels, eval_features, seed)
    if regime == "converged":
        if arm_id not in RESIDUAL_ARMS:
            raise ValueError("the converged regime retrains only the residual arms")
        candidate = {"training": LBFGS, "weight_decay": float(settings["weight_decay"])}
        prediction, _converged = abm_candidate(
            contract, arm_id, settings, candidate, features, labels, eval_features, seed, log, acting=True
        )
        return prediction
    if regime == "tuned":
        return tuned_fit(contract, arm_id, settings, features, labels, agent_ids, eval_features, seed, log)
    raise ValueError(f"unknown regime: {regime}")


class TrainingLearner(abm2.ProbedLearner):
    """The v2 probed learner with only its training replaced by the regime's."""

    regime = "registered"
    contract: Mapping[str, Any] = {}
    created: list[Any] = []

    def __post_init__(self) -> None:
        super().__post_init__()
        self.log = new_log()
        type(self).created.append(self)

    def update(self, events: Sequence[Any]) -> None:
        # The v2 update, line for line, with abm1.fit_batch replaced by train_abm.
        events = [event for event in events if event.agent_id not in self.excluded_agent_ids]
        if not events:
            return
        for event in events:
            self.features.append(event.features)
            self.labels.append(1.0 if event.productive else 0.0)
            self.trained_agent_ids.append(int(event.agent_id))
        seed = int(np.random.SeedSequence((*self.seed_components, self.refits)).generate_state(1)[0])
        self.refits += 1
        prediction = train_abm(
            type(self).contract,
            type(self).regime,
            self.arm_id,
            self.settings,
            np.stack(self.features),
            np.asarray(self.labels, dtype=np.float64),
            np.asarray(self.trained_agent_ids, dtype=np.int64),
            np.vstack([self.eval_features, self.probe_features]),
            seed,
            self.log,
        )
        count = self.eval_features.shape[0]
        self.output = prediction[:count]
        self.probe_output = prediction[count:]


def training_learner(regime: str, contract: Mapping[str, Any]) -> type:
    return type(
        "TaskTrainingLearner",
        (TrainingLearner,),
        {"regime": regime, "contract": contract, "created": []},
    )


@contextlib.contextmanager
def learner_slot(learner_class: type) -> Iterator[None]:
    abm2.ProbedLearner = learner_class
    try:
        yield
    finally:
        abm2.ProbedLearner = ORIGINAL_LEARNER


def policy_candidates(contract: Mapping[str, Any], regime: str) -> list[dict[str, Any]]:
    registered = v3._model_contract(v3.load_contract(), "matched_linear")["candidates"]
    if regime == "registered":
        return [{"training": ADAM, "weight_decay": float(row["weight_decay"])} for row in registered]
    if regime == "converged":
        return [{"training": LBFGS, "weight_decay": float(row["weight_decay"])} for row in registered]
    if regime == "tuned":
        return [dict(row) for row in contract["tuning"]["residual_menu"]]
    raise ValueError(f"unknown regime: {regime}")


def policy_contract(contract: Mapping[str, Any], regime: str) -> dict[str, Any]:
    modified = copy.deepcopy(v3.load_contract())
    for model in modified["model_ladder"]:
        if model["id"] in RESIDUAL_MODELS:
            model["candidates"] = policy_candidates(contract, regime)
    return modified


@contextlib.contextmanager
def candidate_slot(contract: Mapping[str, Any], log: dict[str, Any]) -> Iterator[None]:
    """v3's candidate fit and selection, with converged training and unavailable candidates.

    v3.fit_predict fits every candidate on the inner folds, selects, then makes
    exactly one acting fit. An unconverged inner fit returns NaN, so its mean
    validation Brier is NaN and the selection skips it. The fit after a
    selection is the acting fit, which keeps its final iterate.
    """

    state = {"acting": False}

    def fit_candidate(
        model: Mapping[str, Any],
        candidate: Mapping[str, Any],
        train_features: np.ndarray,
        train_labels: np.ndarray,
        eval_features: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        acting = state["acting"]
        state["acting"] = False
        if candidate.get("training") != LBFGS:
            return ORIGINAL_FIT_CANDIDATE(model, candidate, train_features, train_labels, eval_features, seed)
        kind = "linear" if model["id"] == "matched_linear" else "mlp"
        prediction, info = converge(
            residual_policy(kind, model.get("hidden_width"), seed),
            train_features,
            train_labels,
            eval_features,
            weight_decay=float(candidate["weight_decay"]),
            spec=contract["convergence"],
        )
        _record_lbfgs(log, info, acting=acting)
        if not info["converged"] and not acting:
            return np.full(eval_features.shape[0], np.nan)
        return prediction

    def select_candidate(validation_briers: Sequence[float]) -> int:
        values = np.asarray(validation_briers, dtype=np.float64)
        unavailable = np.isnan(values)
        log["unavailable_candidates"] += int(unavailable.sum())
        state["acting"] = True
        return ORIGINAL_SELECT_CANDIDATE(np.where(unavailable, np.inf, values))

    v3.fit_candidate = fit_candidate
    v3.select_candidate = select_candidate
    try:
        yield
    finally:
        v3.fit_candidate = ORIGINAL_FIT_CANDIDATE
        v3.select_candidate = ORIGINAL_SELECT_CANDIDATE


# -- tasks -------------------------------------------------------------------------------


def _check_slots() -> None:
    if (
        abm2.ProbedLearner is not ORIGINAL_LEARNER
        or v3.fit_candidate is not ORIGINAL_FIT_CANDIDATE
        or v3.select_candidate is not ORIGINAL_SELECT_CANDIDATE
    ):
        raise RuntimeError("a slot was not restored")
    if study2._initial_policy is not ORIGINAL_INITIAL_POLICY:
        raise RuntimeError("the prior slot is not the hand rule")


def _roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def abm_task(args: tuple[Any, ...]) -> dict[str, Any]:
    contract, profile, regime, replicate, visibility_id, world_id, arm_id, evidence_regime = args
    v2_contract = abm2.load_contract()
    config = abm2.study_config(v2_contract, profile)
    probe = abm2.probe_cohort(v2_contract, replicate, world_id)
    common = {
        "replicate": replicate,
        "visibility_id": visibility_id,
        "world_id": world_id,
        "arm_id": arm_id,
        "evidence_regime": evidence_regime,
    }

    def run() -> dict[str, Any]:
        return abm2.run_arm(v2_contract, config, probe=probe, **common)

    if arm_id == "fixed_rule":
        if regime != "registered":
            raise ValueError("the fixed rule runs only in the registered regime")
        result, training = run(), merge_logs([])
    else:
        learner_class = training_learner(regime, contract)
        with learner_slot(learner_class):
            result = run()
        training = merge_logs([learner.log for learner in learner_class.created])
    _check_slots()
    row = {**common, **result}
    parity = None
    if profile == "smoke" and regime == "registered":
        parity = _roundtrip(row) == _roundtrip({**common, **run()})
        _check_slots()
    return {"regime": regime, **row, "training": training, "smoke_parity": parity}


def policy_task(args: tuple[Any, ...]) -> dict[str, Any]:
    contract, profile, regime, replicate = args
    log = new_log()
    with candidate_slot(contract, log):
        rows, _ = v3.run_replicate(policy_contract(contract, regime), profile, replicate)
    _check_slots()
    parity = None
    if profile == "smoke" and regime == "registered":
        unmodified, _ = v3.run_replicate(v3.load_contract(), profile, replicate)
        parity = _roundtrip(rows) == _roundtrip(unmodified)
        _check_slots()
    return {
        "regime": regime,
        "replicate": replicate,
        "rows": [{"regime": regime, **row} for row in rows],
        "training": merge_logs([log]),
        "smoke_parity": parity,
    }


def run_task(task: tuple[str, tuple[Any, ...]]) -> dict[str, Any]:
    kind, args = task
    return abm_task(args) if kind == "abm" else policy_task(args)


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def run_tasks(tasks: Sequence[Any], workers: int, label: str) -> list[Any]:
    """Run tasks in order of submission, longest first, and return them in task order."""

    started = time.monotonic()
    results: list[Any] = [None] * len(tasks)
    if workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            for index, task in enumerate(tasks):
                results[index] = run_task(task)
        return results
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        futures = {pool.submit(run_task, task): index for index, task in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if done % max(1, len(tasks) // 50) == 0 or done == len(tasks):
                elapsed = time.monotonic() - started
                print(f"[{label}] {done}/{len(tasks)} tasks, {elapsed:.0f} s", file=sys.stderr, flush=True)
    return results


def _cost_rank(task: tuple[str, tuple[Any, ...]]) -> tuple[int, ...]:
    """Longest first: tuned field MLP runs, immediate visibility before the lags."""

    kind, args = task
    if kind == "policy":
        return (1, 0, 0)
    _contract, _profile, regime, _replicate, visibility, _world, arm_id, evidence = args
    if (regime, arm_id, evidence) == ("tuned", "mlp_batch", "field_public"):
        return (0, 0 if visibility == "immediate" else 1, 0)
    if evidence == "neighbor_local" and regime == "converged":
        return (2, 0, 0)
    return (
        3,
        0 if arm_id == "mlp_batch" else 1 if arm_id == "tree_batch" else 2,
        0 if regime == "tuned" else 1,
    )


def plan_tasks(contract: Mapping[str, Any], profile: str, regimes: Sequence[str]) -> list[tuple[str, tuple[Any, ...]]]:
    abm_config = abm2.study_config(abm2.load_contract(), profile)
    policy_profile = v2._profile(v3.load_contract(), profile)
    tasks: list[tuple[str, tuple[Any, ...]]] = []
    for regime in regimes:
        tasks.extend(
            ("policy", (contract, profile, regime, replicate))
            for replicate in range(policy_profile["replicate_count"])
        )
        tasks.extend(
            ("abm", (contract, profile, regime, replicate, visibility, world, arm_id, evidence))
            for replicate in range(abm_config.replicates)
            for visibility in abm2.VISIBILITY_IDS
            for world in abm2.FIT_WORLDS
            for arm_id, evidence in REGIME_ARM_KEYS[regime]
        )
    return sorted(tasks, key=_cost_rank)


def _row_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        REGIMES.index(row["regime"]),
        int(row["replicate"]),
        abm2.VISIBILITY_IDS.index(row["visibility_id"]),
        abm2.FIT_WORLDS.index(row["world_id"]),
        ARM_KEYS.index((row["arm_id"], row["evidence_regime"])),
    )


def _policy_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        REGIMES.index(row["regime"]),
        int(row["replicate"]),
        v3.WORLD_IDS.index(row["world_id"]),
        v3.MODEL_IDS.index(row["model_id"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_row(row: Mapping[str, Any]) -> Any:
    return _roundtrip({k: v for k, v in row.items() if k not in ("regime", "training", "smoke_parity")})


def _recorded_rows(contract: Mapping[str, Any], study: str) -> list[dict[str, Any]]:
    spec = contract["studies"][study]
    path = Path(spec["recorded_rows"])
    if _sha256(path) != spec["recorded_rows_sha256"]:
        raise RuntimeError(f"the recorded {study} rows do not match their hash")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_parity(
    contract: Mapping[str, Any],
    profile: str,
    abm_rows: Sequence[Mapping[str, Any]],
    policy_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The registered regime must reproduce its parity reference row for row."""

    registered = [row for row in abm_rows if row["regime"] == "registered"]
    policy_rows = [row for result in policy_results if result["regime"] == "registered" for row in result["rows"]]
    if profile == "smoke":
        abm_ok = all(row["smoke_parity"] for row in registered)
        policy_ok = all(result["smoke_parity"] for result in policy_results if result["regime"] == "registered")
        reference = "THE_UNMODIFIED_RUNNERS"
    else:
        recorded = {tuple(row[key] for key in ROW_KEYS): row for row in _recorded_rows(contract, "abm")}
        config = abm2.study_config(abm2.load_contract(), profile)
        expected = len(ARM_KEYS) * config.replicates * len(abm2.VISIBILITY_IDS) * len(abm2.FIT_WORLDS)
        abm_ok = len(registered) == expected and all(
            _strip_row(row) == recorded[tuple(row[key] for key in ROW_KEYS)] for row in registered
        )
        recorded_policy = {
            (row["replicate"], row["world_id"], row["model_id"]): row
            for row in _recorded_rows(contract, "policy_layer")
        }
        policy_ok = len(policy_rows) == len(recorded_policy) and all(
            _roundtrip({k: v for k, v in row.items() if k != "regime"})
            == recorded_policy[(row["replicate"], row["world_id"], row["model_id"])]
            for row in policy_rows
        )
        reference = "THE_RECORDED_ROWS"
    if not (abm_ok and policy_ok):
        raise RuntimeError(f"registered parity failed: abm={abm_ok}, policy_layer={policy_ok}")
    return {
        "reference": reference,
        "abm_rows": len(registered),
        "policy_layer_rows": len(policy_rows),
        "row_for_row": True,
    }


def run_study(contract: Mapping[str, Any], profile: str, workers: int) -> dict[str, Any]:
    # The registered regime runs first, so a parity failure stops before the long regimes.
    first = run_tasks(plan_tasks(contract, profile, ("registered",)), workers, "registered")
    abm_rows = [result for result in first if "rows" not in result]
    policy_results = [result for result in first if "rows" in result]
    parity = check_parity(contract, profile, abm_rows, policy_results)
    rest = run_tasks(plan_tasks(contract, profile, ("converged", "tuned")), workers, "converged and tuned")
    abm_rows += [result for result in rest if "rows" not in result]
    policy_results += [result for result in rest if "rows" in result]
    for row in abm_rows:
        del row["smoke_parity"]
    return {
        "abm_rows": sorted(abm_rows, key=_row_order),
        "policy_rows": sorted((row for result in policy_results for row in result["rows"]), key=_policy_order),
        "policy_training": {
            regime: merge_logs([r["training"] for r in policy_results if r["regime"] == regime]) for regime in REGIMES
        },
        "parity": parity,
    }


# -- decisions ---------------------------------------------------------------------------


def interval(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    mean = float(np.mean(values))
    half = float(student_t.ppf(0.975, count - 1) * np.std(values, ddof=1) / math.sqrt(count))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": count}


def regime_rows(abm_rows: Sequence[Mapping[str, Any]], regime: str) -> list[Mapping[str, Any]]:
    registered = [row for row in abm_rows if row["regime"] == "registered"]
    own = [row for row in abm_rows if row["regime"] == regime]
    if regime == "registered":
        return registered
    if regime == "converged":
        return [row for row in registered if row["arm_id"] not in RESIDUAL_ARMS] + own
    if regime == "tuned":
        return [row for row in registered if row["arm_id"] == "fixed_rule"] + own
    raise ValueError(f"unknown regime: {regime}")


def _values(rows: Sequence[Mapping[str, Any]], *, visibility: str, world: str, arm: str, evidence: str, metric: str) -> np.ndarray:
    chosen = sorted(
        (row for row in rows if (row["visibility_id"], row["world_id"], row["arm_id"], row["evidence_regime"]) == (visibility, world, arm, evidence)),
        key=lambda row: int(row["replicate"]),
    )
    return np.asarray([float(row[metric]) for row in chosen], dtype=np.float64)


def post_hoc(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    share = {}
    for world in abm2.INTERACTION_FIT_WORLDS:
        gap = _values(rows, visibility="immediate", world=world, arm="mlp_batch", evidence="field_public", metric="productive_share_of_entrants") - _values(
            rows, visibility="immediate", world=world, arm="tree_batch", evidence="field_public", metric="productive_share_of_entrants"
        )
        share[world] = {
            "mean": round(float(gap.mean()), 6),
            "min": round(float(gap.min()), 6),
            "max": round(float(gap.max()), 6),
            "positive_replicates": int(np.sum(gap > 0.0)),
        }
    return {"raw_entry_share_mlp_minus_tree_immediate": share}


def _strip(dispositions: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dispositions.items() if key not in ("reports", "status")}


def resolve(document: Mapping[str, Any], pointer: str) -> Any:
    value: Any = document
    for part in pointer.split("/"):
        value = value[part]
    return value


def check_holds(value: Any, check: Mapping[str, Any]) -> bool:
    if "equals" in check:
        return value == check["equals"]
    if "contains" in check:
        return check["contains"] in value
    if "greater_than" in check:
        return float(value) > float(check["greater_than"])
    if "less_than" in check:
        return float(value) < float(check["less_than"])
    raise ValueError(f"unknown check: {check}")


def evaluate_claims(contract: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    claims = {}
    for claim in contract["decision_contract"]["capacity_claims"]:
        checks = [
            {**check, "observed": resolve(view, check["pointer"]), "holds": check_holds(resolve(view, check["pointer"]), check)}
            for check in claim["checks"]
        ]
        claims[claim["id"]] = {"holds": all(check["holds"] for check in checks), "checks": checks}
    changed = sorted(claim_id for claim_id, row in claims.items() if not row["holds"])
    return {"outcome": "CLAIMS_CHANGE" if changed else "CLAIMS_HOLD", "changed": changed, "claims": claims}


def decide(contract: Mapping[str, Any], study: Mapping[str, Any]) -> dict[str, Any]:
    v2_contract = abm2.load_contract()
    v3_contract = v3.load_contract()
    by_regime: dict[str, Any] = {}
    for regime in REGIMES:
        rows = regime_rows(study["abm_rows"], regime)
        abm = abm2.decide(rows, v2_contract)["by_visibility"]
        policy_rows = [row for row in study["policy_rows"] if row["regime"] == regime]
        policy = v3.decide(policy_rows, v3_contract)
        view = {
            "policy_layer": _strip(policy),
            "abm": {visibility: _strip(row) for visibility, row in abm.items()},
            "post_hoc": post_hoc(rows),
        }
        by_regime[regime] = {
            "dispositions": view,
            "claims": evaluate_claims(contract, view),
            "reports": {
                "policy_layer": policy["reports"],
                "abm": {visibility: row["reports"] for visibility, row in abm.items()},
            },
        }
    linear = {"visibility": "immediate", "world": "linear_fit", "evidence": "field_public", "metric": "probe_brier"}
    edges = {}
    for regime in REGIMES:
        rows = regime_rows(study["abm_rows"], regime)
        edges[regime] = interval(_values(rows, arm="linear_batch", **linear) - _values(rows, arm="mlp_batch", **linear))
    budget = interval(
        _values(regime_rows(study["abm_rows"], "converged"), arm="mlp_batch", **linear)
        - _values(regime_rows(study["abm_rows"], "registered"), arm="mlp_batch", **linear)
    )
    behavior = {
        regime: {
            visibility: {
                "entry": row["entry_passing_family_geometries"],
                "exit": row["exit_passing_family_geometries"],
            }
            for visibility, row in by_regime[regime]["dispositions"]["abm"].items()
        }
        for regime in REGIMES
    }
    holds = {regime: by_regime[regime]["claims"]["outcome"] == "CLAIMS_HOLD" for regime in REGIMES}
    tuned_immediate = behavior["tuned"]["immediate"]
    expectations = {
        "E1": {"held": holds["tuned"], "observed": by_regime["tuned"]["claims"]["changed"]},
        "E2": {"held": edges["tuned"]["mean"] < 0.01, "observed": edges["tuned"]["mean"]},
        "E3": {"held": budget["low"] > 0.0, "observed": budget},
        "E4": {
            "held": any(item.startswith("mlp_batch/") for item in tuned_immediate["entry"] + tuned_immediate["exit"]),
            "observed": tuned_immediate,
        },
    }
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "o36": STUDY_STATUS[(holds["converged"], holds["tuned"])],
        "by_regime": by_regime,
        "registered_quantities": {
            "linear_world_mlp_edge": edges,
            "mlp_budget_effect": budget,
            "behavior_passing_lists": behavior,
        },
        "expectations": expectations,
    }


def check_registered(contract: Mapping[str, Any], decisions: Mapping[str, Any]) -> dict[str, Any]:
    """In development the registered regime must restate both records and every claim."""

    registered = decisions["by_regime"]["registered"]
    abm_record = json.loads((ROOT / contract["studies"]["abm"]["recorded_summary"]).read_text())
    policy_record = json.loads((ROOT / contract["studies"]["policy_layer"]["recorded_summary"]).read_text())
    same_abm = _roundtrip(registered["dispositions"]["abm"]) == abm_record["registered_result_disposition"]
    same_policy = _roundtrip(registered["dispositions"]["policy_layer"]) == policy_record["registered_result_disposition"]
    claims = registered["claims"]["outcome"] == "CLAIMS_HOLD"
    if not (same_abm and same_policy and claims):
        raise RuntimeError(
            f"registered regime check failed: abm={same_abm}, policy_layer={same_policy}, claims={claims}"
        )
    return {"dispositions_equal_both_records": True, "every_claim_holds": True}


# -- summary -----------------------------------------------------------------------------


def convergence_check(contract: Mapping[str, Any], training: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Unconverged acting fits, per study and regime, against the registered ceiling."""

    ceiling = float(contract["convergence"]["non_converged"]["acting_ceiling_share"])
    rows = {}
    for study_name, by_regime in training.items():
        for regime, log in by_regime.items():
            acting = int(log["lbfgs_acting_fits"])
            unconverged = int(log["lbfgs_unconverged_acting"])
            rows[f"{study_name}/{regime}"] = {
                "lbfgs_acting_fits": acting,
                "unconverged_acting_fits": unconverged,
                "unconverged_acting_share": unconverged / acting if acting else 0.0,
                "lbfgs_inner_fits": int(log["lbfgs_fits"]) - acting,
                "unconverged_inner_fits": int(log["lbfgs_unconverged_inner"]),
                "unavailable_candidates": int(log["unavailable_candidates"]),
                "max_acting_gradient": float(log["max_acting_gradient"]),
            }
    return {
        "acting_ceiling_share": ceiling,
        "within_ceiling": all(row["unconverged_acting_share"] <= ceiling for row in rows.values()),
        "by_study_and_regime": rows,
    }


CELL_METRICS = ("probe_brier", "probe_auc", "productive_share_of_entrants", "active_failed_rate", "cumulative_entry_rate")


def summarize(contract: Mapping[str, Any], study: Mapping[str, Any], profile: str) -> dict[str, Any]:
    cells = []
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in study["abm_rows"]:
        grouped[(row["regime"], row["visibility_id"], row["world_id"], row["arm_id"], row["evidence_regime"])].append(row)
    for key, rows in grouped.items():
        means = {}
        for metric in CELL_METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            means[metric] = round(float(np.mean(values)), 6) if values else None
        cells.append(
            {
                "regime": key[0],
                "visibility_id": key[1],
                "world_id": key[2],
                "arm_id": key[3],
                "evidence_regime": key[4],
                "replicates": len(rows),
                "means": means,
                "training": merge_logs([row["training"] for row in rows]),
            }
        )
    policy_cells = []
    policy_grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in study["policy_rows"]:
        policy_grouped[(row["regime"], row["world_id"], row["model_id"])].append(row)
    for (regime, world, model), rows in policy_grouped.items():
        selected = Counter(row["selected_candidate"] for row in rows if "selected_candidate" in row)
        policy_cells.append(
            {
                "regime": regime,
                "world_id": world,
                "model_id": model,
                "replicates": len(rows),
                "brier_score": round(float(np.mean([row["brier_score"] for row in rows])), 6),
                "roc_auc": round(float(np.mean([row["roc_auc"] for row in rows])), 6),
                "selected_candidates": {str(k): v for k, v in sorted(selected.items())},
            }
        )
    training = {
        regime: merge_logs([row["training"] for row in study["abm_rows"] if row["regime"] == regime])
        for regime in REGIMES
    }
    registered_neighbor = {
        tuple(row[key] for key in ROW_KEYS): _strip_row(row)
        for row in study["abm_rows"]
        if row["regime"] == "registered" and row["evidence_regime"] == "neighbor_local"
    }
    tuned_neighbor = [
        row for row in study["abm_rows"] if row["regime"] == "tuned" and row["evidence_regime"] == "neighbor_local"
    ]
    neighbor_equal = all(
        _strip_row(row) == registered_neighbor[tuple(row[key] for key in ROW_KEYS)] for row in tuned_neighbor
    )
    convergence = convergence_check(contract, {"abm": training, "policy_layer": study["policy_training"]})
    replicates = len({row["replicate"] for row in study["abm_rows"]})
    if replicates < int(contract["decision_contract"]["minimum_replicates"]):
        decisions: dict[str, Any] = {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        registered_check: dict[str, Any] = {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
    elif not convergence["within_ceiling"]:
        # The rows are kept; no regime is judged when too many acting fits missed the tolerance.
        decisions = {"status": "NOT_EVALUATED_ACTING_CONVERGENCE_ABOVE_THE_CEILING"}
        registered_check = {"status": "NOT_EVALUATED_ACTING_CONVERGENCE_ABOVE_THE_CEILING"}
    else:
        decisions = decide(contract, study)
        registered_check = check_registered(contract, decisions)
    return {
        "schema_version": "pivot_converged_capacity_scorecard.v2",
        "status": "SYNTHETIC_TRAINING_REGIME_COMPARISON",
        "profile": profile,
        "checks": {
            "parity": study["parity"],
            "registered": registered_check,
            "slots_restored": True,
            "convergence": convergence,
        },
        "training": {"abm": training, "policy_layer": study["policy_training"]},
        "tuned_neighbor_rows_equal_registered": neighbor_equal,
        "cells": sorted(cells, key=lambda c: (REGIMES.index(c["regime"]), c["visibility_id"], c["world_id"], c["arm_id"], c["evidence_regime"])),
        "policy_cells": sorted(policy_cells, key=lambda c: (REGIMES.index(c["regime"]), c["world_id"], c["model_id"])),
        "abm_row_count": len(study["abm_rows"]),
        "policy_row_count": len(study["policy_rows"]),
        "decisions": decisions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot converged capacity comparison scorecard v2",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| Regime | Visibility | World | Arm | Evidence | Probe Brier | Entrant share | Active failed |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        m = cell["means"]
        lines.append(
            f"| {cell['regime']} | {cell['visibility_id']} | {cell['world_id']} | {cell['arm_id']} | "
            f"{cell['evidence_regime']} | {m['probe_brier']:.4f} | {m['productive_share_of_entrants']:.4f} | "
            f"{m['active_failed_rate']:.4f} |"
        )
    lines.extend(["", "| Regime | World | Model | Brier | AUC |", "|---|---|---|---:|---:|"])
    for cell in summary["policy_cells"]:
        lines.append(
            f"| {cell['regime']} | {cell['world_id']} | {cell['model_id']} | {cell['brier_score']:.4f} | {cell['roc_auc']:.4f} |"
        )
    decisions = summary["decisions"]
    lines.extend(["", f"Decision status: `{decisions['status']}`", ""])
    if decisions["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE":
        lines.append(f"- O36: `{decisions['o36']}`")
        for regime, row in decisions["by_regime"].items():
            changed = ", ".join(row["claims"]["changed"]) or "none"
            lines.append(f"- {regime}: `{row['claims']['outcome']}`; changed: {changed}")
        for key, row in decisions["expectations"].items():
            lines.append(f"- {key}: {'held' if row['held'] else 'failed'}")
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    study: Mapping[str, Any],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in study["abm_rows"]),
        "policy_rows.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in study["policy_rows"]),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in files.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_converged_capacity_receipt.v2",
        "status": "COMPLETE",
        "abm_row_count": len(study["abm_rows"]),
        "policy_row_count": len(study["policy_rows"]),
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {name: _sha256(output_dir / name) for name in files},
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    tasks = {profile: len(plan_tasks(contract, profile, REGIMES)) for profile in ("smoke", "development")}
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": _sha256(config_path),
        "regimes": list(REGIMES),
        "tasks": tasks,
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
    started = time.monotonic()
    study = run_study(contract, args.profile, workers=args.workers)
    summary = {
        **summarize(contract, study, args.profile),
        "precommit_sha": args.precommit_sha,
    }
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        study=study,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "abm_row_count": receipt["abm_row_count"],
                "policy_row_count": receipt["policy_row_count"],
                "wall_seconds": round(time.monotonic() - started, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
