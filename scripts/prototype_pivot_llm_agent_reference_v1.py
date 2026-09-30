#!/usr/bin/env python3
"""Run the precommitted Pivot LLM-agent reference v1 (N06).

A local LLM is queried at every agent-step as the agent's policy, inside the
same lifecycle as the NABM arms. ``run`` needs transformers and a GPU:

    uv run --with 'transformers==5.14.1' --with accelerate \
        python scripts/prototype_pivot_llm_agent_reference_v1.py run ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import platform
import re
import sys
import time
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
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_agent_reference_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_prior_pilot_v4 as pilot4  # noqa: E402


pilot3 = pilot4.pilot3
pilot1 = pilot4.pilot1
abm1 = pilot1.abm1
abm2 = pilot1.abm2
delayed = pilot1.delayed
persistence = pilot1.persistence
study2 = pilot1.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = pilot1.FIT_WORLDS
FEATURES = pilot1.FEATURES
NABM_ARMS = pilot3.ABM_ARMS
FIELD_ARMS = pilot3.FIELD_ARMS
LLM_ARM = ("llm_agent", "field_public_in_context")
NABM_CONDITIONS = ("hand", "hybrid_qwen3_4b_2507")
LLM_CONDITION = "llm_agent_qwen3_4b_2507"
CONDITION_IDS = (*NABM_CONDITIONS, LLM_CONDITION)
ORIGINAL_AGENT = study2.PivotPolicyAgent
OPENED_SEEDS = pilot4.OPENED_SEEDS | {20261025, 20261026}
V1_DESIGN_SEED = pilot4.V1_DESIGN_SEED
NONINFERIORITY_MARGIN = pilot3.NONINFERIORITY_MARGIN


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("LLM-agent reference contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_agent_reference.v1":
        raise ValueError("unexpected LLM-agent reference schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("LLM-agent reference contract is not a precommit design")
    if [row["id"] for row in contract["conditions"]] != list(CONDITION_IDS):
        raise ValueError("conditions changed")
    share = float(contract["exploration_floor"]["reference_share"])
    for row in contract["conditions"]:
        expected = 0.0 if row["id"] == "hand" else share
        if float(row["exploratory_share"]) != expected:
            raise ValueError(f"{row['id']} must use exploratory share {expected}")
    if [tuple(row) for row in contract["abm_arms"]] != list(NABM_ARMS):
        raise ValueError("NABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if int(contract["elicitation"]["design_cohort"]["base_seed"]) != V1_DESIGN_SEED:
        raise ValueError("calibration must use the v1 design cohort")
    if int(contract["runtime"]["batch_size"]) != 1:
        raise ValueError("agent queries run alone so no answer depends on its batch")
    config = contract["profiles"]["development"]
    open_steps = int(config["steps"]) - int(config["burn_in_steps"])
    if int(contract["exploration_floor"]["open_steps"]) != open_steps:
        raise ValueError("exploration open steps must match the development profile")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != pilot1.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this study")
    seeds = {int(config["base_seed"]), int(contract["probe_cohort"]["base_seed"])}
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("the LLM-agent reference needs fresh seed namespaces")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    if contract["boundaries"].get("paid_api_calls") != 0:
        raise ValueError("the LLM-agent reference makes no paid API calls")


def _condition(contract: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    return next(row for row in contract["conditions"] if row["id"] == condition_id)


def load_v4_prior(contract: Mapping[str, Any]) -> dict[str, Any]:
    spec = contract["v4_prior"]
    path = pilot4._bound(spec["scorecard"], spec["sha256"], "the v4 scorecard")
    prior = json.loads(path.read_text(encoding="utf-8"))["calibrated_priors"][spec["model"]]
    return {"weights": [float(v) for v in prior["weights"]], "bias": float(prior["bias"])}


# -- evidence and prompts ------------------------------------------------------------


def evidence_text(
    contract: Mapping[str, Any],
    events: Sequence[Any],
    features: np.ndarray,
    exclude_agent_id: int | None,
) -> str:
    """Field-wide public outcome counts plus those of the most similar profiles."""

    spec = contract["agent_prompt"]
    visible = [event for event in events if event.agent_id != exclude_agent_id]
    if not visible:
        return spec["no_evidence_text"]
    productive = sum(bool(event.productive) for event in visible)
    stacked = np.stack([event.features for event in visible])
    distance = np.sqrt(np.sum((stacked - features) ** 2, axis=1))
    ids = np.asarray([event.agent_id for event in visible])
    order = np.lexsort((ids, distance))[: int(spec["similar_outcomes"])]
    similar = [visible[index] for index in order]
    similar_productive = sum(bool(event.productive) for event in similar)
    return spec["evidence_template"].format(
        productive=productive,
        failed=len(visible) - productive,
        similar=len(similar),
        similar_productive=similar_productive,
        similar_failed=len(similar) - similar_productive,
    )


def agent_prompt(contract: Mapping[str, Any], features: np.ndarray, evidence: str) -> str:
    elicitation = contract["elicitation"]
    profile = elicitation["profile_template"].format(
        **{name: float(value) for name, value in zip(FEATURES, features, strict=True)}
    )
    return f"{profile}\n\n{evidence}\n\n{elicitation['primary_question']}"


# -- LLM reader ---------------------------------------------------------------------


class LLMReader:
    """Prefilled forced-choice log-odds, one prompt at a time, with an exact cache."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        import transformers  # noqa: PLC0415

        spec = contract["llm"]
        runtime = contract["runtime"]
        if transformers.__version__ != runtime["transformers"]:
            raise RuntimeError(f"transformers {transformers.__version__} is not pinned")
        cached = try_to_load_from_cache(spec["model_id"], "config.json", revision=spec["snapshot"])
        if not isinstance(cached, str):
            raise RuntimeError("the pinned model snapshot is not in the local cache")
        snapshot_dir = Path(cached).parent
        for name, expected in spec["snapshot_file_sha256"].items():
            pilot4._bound(snapshot_dir / name, expected, f"model snapshot file {name}")
        self.weight_shard_sha256 = {
            path.name: pilot1._sha256(path) for path in sorted(snapshot_dir.glob("*.safetensors"))
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["snapshot"], local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"],
            revision=spec["snapshot"],
            local_files_only=True,
            dtype=getattr(torch, runtime["dtype"]),
            device_map=runtime["device"],
        ).eval()
        self.device = runtime["device"]
        self.system = contract["elicitation"]["system_prompt"]
        self.template_kwargs = dict(spec["chat_template_kwargs"])
        self.prefill = contract["elicitation"]["readout"]["assistant_prefill"]
        self.yes_id, self.no_id = pilot1._answer_token_ids(
            self.tokenizer, contract["elicitation"]["answer_tokens"]
        )
        rendered = pilot4.render_chat(self.tokenizer, self.system, ["x"], self.template_kwargs)[0]
        if not rendered.endswith(spec["generation_prompt_suffix"]):
            raise RuntimeError("the chat template does not end in the registered suffix")
        self.cache: dict[str, tuple[float, float]] = {}
        self.forward_passes = 0
        self.gpu_seconds = 0.0
        self.environment = {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
        }

    def _forward(self, prompt: str) -> tuple[float, float]:
        text = pilot4.render_chat(self.tokenizer, self.system, [prompt], self.template_kwargs)[0]
        batch = self.tokenizer([text + self.prefill], return_tensors="pt").to(self.device)
        torch.cuda.synchronize()
        began = time.time()
        with torch.no_grad():
            logits = self.model(**batch, logits_to_keep=1).logits[0, -1, :].float()
        torch.cuda.synchronize()
        self.gpu_seconds += time.time() - began
        self.forward_passes += 1
        probabilities = torch.softmax(logits, dim=-1)
        odds = float(logits[self.yes_id] - logits[self.no_id])
        mass = float(probabilities[self.yes_id] + probabilities[self.no_id])
        return odds, mass

    def log_odds(self, prompts: Sequence[str]) -> np.ndarray:
        values = []
        for prompt in prompts:
            if prompt not in self.cache:
                self.cache[prompt] = self._forward(prompt)
            values.append(self.cache[prompt][0])
        return np.asarray(values, dtype=np.float64)

    def answer_mass(self, prompts: Sequence[str]) -> np.ndarray:
        self.log_odds(prompts)
        return np.asarray([self.cache[prompt][1] for prompt in prompts])


def calibration_shift(log_odds: np.ndarray, target: float, interval: Sequence[float], tolerance: float) -> float:
    """Aggregate-only calibration: shift log-odds until the mean probability hits the target."""

    def mean(shift: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(log_odds + shift)))))

    low, high = float(interval[0]), float(interval[1])
    if not mean(low) < target < mean(high):
        raise RuntimeError("the calibration target is outside the bisection interval")
    for _iteration in range(400):
        middle = 0.5 * (low + high)
        value = mean(middle)
        if abs(value - target) < tolerance:
            return middle
        if value < target:
            low = middle
        else:
            high = middle
    raise RuntimeError("calibration did not converge")


def phase_a(contract: Mapping[str, Any], reader: Any) -> dict[str, Any]:
    features, _keys = pilot1.design_cohort(contract)
    no_evidence = contract["agent_prompt"]["no_evidence_text"]
    prompts = [agent_prompt(contract, row, no_evidence) for row in features]
    odds = reader.log_odds(prompts)
    repeat_count = int(contract["runtime"]["determinism_repeat_prompts"])
    repeated = np.asarray([reader._forward(prompt)[0] for prompt in prompts[:repeat_count]])
    calibration = contract["calibration"]
    shift = calibration_shift(
        odds,
        float(calibration["stated_population_base_rate"]),
        calibration["shift_interval"],
        float(calibration["tolerance"]),
    )
    mass = reader.answer_mass(prompts)
    return {
        "profiles": len(prompts),
        "calibration_shift": shift,
        "achieved_mean": float(np.mean(1.0 / (1.0 + np.exp(-(odds + shift))))),
        "raw_mean_probability": float(np.mean(1.0 / (1.0 + np.exp(-odds)))),
        "validity": {
            "median_answer_token_mass": float(np.median(mass)),
            "weak_format": bool(np.median(mass) < 0.5),
        },
        "determinism": {
            "repeat_prompts": repeat_count,
            "repeat_identical": bool(np.array_equal(odds[:repeat_count], repeated)),
        },
    }


# -- the LLM agent -------------------------------------------------------------------


class LLMField:
    """Computes every agent's policy output once per step from the public record."""

    def __init__(self, contract: Mapping[str, Any], policy: Any, shift: float) -> None:
        self.contract = contract
        self.policy = policy
        self.shift = float(shift)
        self.agents: list[Any] = []
        self.environment: Any = None
        self.step_computed: int | None = None
        self.outputs: dict[int, float] = {}
        self.agent_steps_queried = 0

    def register(self, agent: Any) -> None:
        self.agents.append(agent)
        self.environment = agent.environment

    def output_for(self, agent: Any) -> float:
        step = int(self.environment.step)
        if self.step_computed != step:
            self._compute(step, self.agents)
        elif agent.agent_id not in self.outputs:
            # An agent built after this step's outputs were computed.
            self._compute(step, [agent])
        return self.outputs[agent.agent_id]

    def _needs_query(self, agent: Any) -> bool:
        base = agent.base
        return agent.agent_id not in self.outputs or not base.pivoted or not base.productive_pivot

    def _compute(self, step: int, agents: Sequence[Any]) -> None:
        events = list(self.environment.public_events)
        queried = [agent for agent in agents if self._needs_query(agent)]
        prompts = [
            agent_prompt(
                self.contract,
                agent.features,
                evidence_text(self.contract, events, agent.features, agent.agent_id),
            )
            for agent in queried
        ]
        odds = self.policy.log_odds(prompts) if prompts else np.zeros(0)
        for agent, value in zip(queried, odds, strict=True):
            self.outputs[agent.agent_id] = float(1.0 / (1.0 + np.exp(-(value + self.shift))))
        self.agent_steps_queried += len(queried)
        self.step_computed = step

    def probe_predictions(self, probe_features: np.ndarray) -> np.ndarray:
        events = list(self.environment.public_events)
        prompts = [
            agent_prompt(self.contract, row, evidence_text(self.contract, events, row, None))
            for row in probe_features
        ]
        odds = self.policy.log_odds(prompts)
        return 1.0 / (1.0 + np.exp(-(odds + self.shift)))


class HandField(LLMField):
    """The LLM-agent lifecycle with the agent's own hand-prior policy, for nesting."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        super().__init__(contract, None, 0.0)

    def _compute(self, step: int, agents: Sequence[Any]) -> None:
        for agent in agents:
            self.outputs[agent.agent_id] = ORIGINAL_AGENT.policy_output(agent, agent.features)
        self.step_computed = step


def llm_agent_class(field: LLMField) -> type:
    def __post_init__(self: Any) -> None:
        self.features = study2._feature_vector(self.base)
        self.prior_weight = self.weight.detach().clone()
        self.prior_bias = self.bias.detach().clone()
        field.register(self)
        self.last_policy_output = self.policy_output(self.features)

    def policy_output(self: Any, features: np.ndarray) -> float:
        del features
        return field.output_for(self)

    def act_or_predict(self: Any, observation: Any) -> torch.Tensor:
        del observation
        return torch.tensor([self.policy_output(self.features)], dtype=torch.float64)

    return type(
        "LLMPolicyAgent",
        (ORIGINAL_AGENT,),
        {"__post_init__": __post_init__, "policy_output": policy_output, "act_or_predict": act_or_predict},
    )


@contextlib.contextmanager
def agent_slot(agent_class: type) -> Iterator[None]:
    study2.PivotPolicyAgent = agent_class
    try:
        yield
    finally:
        study2.PivotPolicyAgent = ORIGINAL_AGENT


def run_llm_agent(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    world_id: str,
    field: LLMField,
) -> dict[str, Any]:
    """One LLM-agent run through the fixed-rule lifecycle with the LLM as the policy."""

    config = abm1.study_config(contract, profile)
    condition = _condition(contract, LLM_CONDITION)
    epsilon = pilot3.epsilon_for(float(condition["exploratory_share"]), config.steps - config.burn_in_steps)
    probe_features, probe_labels = abm2.probe_cohort(contract, replicate, world_id)
    adapter_class = pilot3.exploration_adapter(epsilon, config.base_seed, replicate)
    with agent_slot(llm_agent_class(field)), pilot3.adapter_slot(adapter_class):
        run = abm2.run_arm(
            contract,
            config,
            replicate=replicate,
            visibility_id="immediate",
            world_id=world_id,
            arm_id="fixed_rule",
            evidence_regime="none",
            probe=(probe_features, probe_labels),
        )
    if study2.PivotPolicyAgent is not ORIGINAL_AGENT:
        raise RuntimeError("the agent slot was not restored")
    if persistence.PersistenceCommitAdapter is not pilot3.ORIGINAL_ADAPTER:
        raise RuntimeError("the adapter slot was not restored")
    if len(adapter_class.created) != 1 or len(field.agents) != config.agent_count:
        raise RuntimeError("the LLM-agent run did not build one adapter and one agent per researcher")
    predictions = field.probe_predictions(probe_features)
    return {
        **run,
        "probe_brier": abm2._brier(predictions, probe_labels),
        "probe_auc": abm2._auc(predictions, probe_labels),
        "evidence_events_per_learner": float(len(field.environment.public_events)),
        "exploratory_entry_rate": adapter_class.created[0].exploratory_entries / config.agent_count,
        "exploratory_share": float(condition["exploratory_share"]),
        "epsilon": epsilon,
        "agent_steps_queried": field.agent_steps_queried,
    }


def check_llm_machinery_nests_the_fixed_rule(
    contract: Mapping[str, Any], profile: str, replicate: int, world_id: str
) -> None:
    """With the hand prior as the policy, the LLM-agent path equals the fixed-rule arm."""

    config = abm1.study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    reference = abm2.run_arm(
        contract, config, replicate=replicate, visibility_id="immediate", world_id=world_id,
        arm_id="fixed_rule", evidence_regime="none", probe=probe,
    )
    field = HandField(contract)
    with agent_slot(llm_agent_class(field)):
        nested = abm2.run_arm(
            contract, config, replicate=replicate, visibility_id="immediate", world_id=world_id,
            arm_id="fixed_rule", evidence_regime="none", probe=probe,
        )
    if study2.PivotPolicyAgent is not ORIGINAL_AGENT:
        raise RuntimeError("the agent slot was not restored")
    for key, value in reference.items():
        if nested[key] != value:
            raise RuntimeError(f"the LLM-agent machinery does not nest the fixed rule at {key}")


# -- NABM conditions (CPU) -------------------------------------------------------------


def nabm_prior_for(
    condition: Mapping[str, Any], v4_prior: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if condition["prior"] == "hand":
        return None
    if condition["prior"] == "v4_qwen3_4b_2507":
        return v4_prior
    raise ValueError(f"unknown NABM prior: {condition['prior']}")


def run_nabm_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    world_id: str,
    condition_id: str,
    v4_prior: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    began = time.process_time()
    config = abm1.study_config(contract, profile)
    condition = _condition(contract, condition_id)
    share = float(condition["exploratory_share"])
    epsilon = pilot3.epsilon_for(share, config.steps - config.burn_in_steps)
    prior = nabm_prior_for(condition, v4_prior)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows, runs = [], {}
    for arm_id, evidence_regime in NABM_ARMS:
        adapter_class = pilot3.exploration_adapter(epsilon, config.base_seed, replicate)
        with pilot1.prior_slot(prior), pilot3.adapter_slot(adapter_class):
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
    if persistence.PersistenceCommitAdapter is not pilot3.ORIGINAL_ADAPTER:
        raise RuntimeError("the adapter slot was not restored")
    if condition_id == "hand" and world_id == "linear_fit":
        pilot3.check_nesting(contract, config, replicate=replicate, fixed_run=runs["fixed_rule"])
    return rows, time.process_time() - began


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_nabm_args(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], float]:
    return run_nabm_task(*args)


def run_nabm(
    contract: Mapping[str, Any], profile: str, prior: Mapping[str, Any], workers: int = 1
) -> tuple[list[dict[str, Any]], list[float]]:
    config = abm1.study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, world_id, condition_id, prior)
        for replicate in range(config.replicates)
        for world_id in FIT_WORLDS
        for condition_id in NABM_CONDITIONS
    ]
    if workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            results = [_run_nabm_args(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_worker_init
        ) as pool:
            results = list(pool.map(_run_nabm_args, tasks))
    rows = [row for task_rows, _seconds in results for row in task_rows]
    seconds = [task_seconds for _rows, task_seconds in results]
    return rows, seconds


def run_llm(
    contract: Mapping[str, Any], profile: str, policy: Any, shift: float
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    config = abm1.study_config(contract, profile)
    rows, timing = [], []
    for replicate in range(config.replicates):
        for world_id in FIT_WORLDS:
            field = LLMField(contract, policy, shift)
            passes = getattr(policy, "forward_passes", 0)
            seconds = getattr(policy, "gpu_seconds", 0.0)
            began = time.time()
            run = run_llm_agent(contract, profile, replicate, world_id, field)
            rows.append(
                {
                    "replicate": replicate,
                    "world_id": world_id,
                    "condition_id": LLM_CONDITION,
                    "arm_id": LLM_ARM[0],
                    "evidence_regime": LLM_ARM[1],
                    **run,
                }
            )
            timing.append(
                {
                    "wall_seconds": time.time() - began,
                    "gpu_seconds": getattr(policy, "gpu_seconds", 0.0) - seconds,
                    "forward_passes": getattr(policy, "forward_passes", 0) - passes,
                }
            )
    return rows, timing


# -- decisions ------------------------------------------------------------------------


def _values(indexed, replicates, world, condition, arm, metric) -> np.ndarray:  # noqa: ANN001
    return pilot3._values(indexed, replicates, world, condition, arm, metric)


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), row["world_id"], row["condition_id"], row["arm_id"]): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    share = "productive_share_of_entrants"
    llm = (LLM_CONDITION, LLM_ARM[0])
    nabm_rule = decision["nabm_versus_llm_agent"]
    nabm_cells, hand_cells = {}, {}
    for world in FIT_WORLDS:
        llm_precision = _values(indexed, replicates, world, *llm, share)
        llm_brier = _values(indexed, replicates, world, *llm, "probe_brier")
        for arm in FIELD_ARMS:
            key = f"{world}/{arm}"
            precision_gap = _values(indexed, replicates, world, NABM_CONDITIONS[1], arm, share) - llm_precision
            learning_gap = llm_brier - _values(indexed, replicates, world, NABM_CONDITIONS[1], arm, "probe_brier")
            precision_ok = float(np.mean(precision_gap)) >= float(nabm_rule["precision_margin"])
            learning_ok = float(np.mean(learning_gap)) >= float(nabm_rule["learning_margin"])
            if precision_ok and learning_ok:
                disposition = "NABM_MATCHES_LLM_AGENT"
            elif learning_ok:
                disposition = "LLM_AGENT_BETTER_PRECISION"
            elif precision_ok:
                disposition = "LLM_AGENT_BETTER_LEARNING"
            else:
                disposition = "LLM_AGENT_BETTER_BOTH"
            nabm_cells[key] = {
                "disposition": disposition,
                "nabm_minus_llm_precision": abm1.v2._distribution(precision_gap),
                "llm_minus_nabm_probe_brier": abm1.v2._distribution(learning_gap),
                "nabm_precision_superior": abm1._judge(
                    precision_gap, float(decision["superiority"]["precision_at_least"]), required
                ),
                "nabm_learning_superior": abm1._judge(
                    learning_gap, float(decision["superiority"]["probe_brier_at_least"]), required
                ),
            }
            hand_precision = _values(indexed, replicates, world, "hand", arm, share)
            precision = abm1._judge(
                llm_precision - hand_precision,
                float(decision["precision_versus_hand"]["improvement_at_least"]),
                required,
            )
            gap = _values(indexed, replicates, world, "hand", arm, "probe_brier") - llm_brier
            noninferior = {
                "hand_minus_llm_probe_brier": abm1.v2._distribution(gap),
                "pass": float(np.mean(gap)) >= NONINFERIORITY_MARGIN,
            }
            if precision["pass"] and noninferior["pass"]:
                hand_disposition = "LLM_AGENT_DOMINATES_HAND"
            elif precision["pass"]:
                hand_disposition = "SELECTION_ONLY"
            elif noninferior["pass"]:
                hand_disposition = "LEARNING_ONLY"
            else:
                hand_disposition = "NEITHER"
            hand_cells[key] = {
                "disposition": hand_disposition,
                "precision_versus_hand": precision,
                "learning_noninferior_versus_hand": noninferior,
            }
    matching = sorted(k for k, v in nabm_cells.items() if v["disposition"] == "NABM_MATCHES_LLM_AGENT")
    dominating = sorted(k for k, v in hand_cells.items() if v["disposition"] == "LLM_AGENT_DOMINATES_HAND")
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "nabm_versus_llm_agent": {
            "summary": pilot4._count_label(len(matching), len(nabm_cells), "NABM_MATCHES_LLM_AGENT"),
            "matching_cells": matching,
            "cells": nabm_cells,
        },
        "llm_agent_versus_hand": {
            "summary": pilot4._count_label(len(dominating), len(hand_cells), "DOMINATES"),
            "dominating_cells": dominating,
            "cells": hand_cells,
        },
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
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
    arms = {condition: NABM_ARMS for condition in NABM_CONDITIONS} | {LLM_CONDITION: (LLM_ARM,)}
    cells = []
    for world in FIT_WORLDS:
        for condition in CONDITION_IDS:
            for arm, evidence in arms[condition]:
                cell_rows = grouped[(world, condition, arm)]
                means = {}
                for metric in metrics:
                    values = [row[metric] for row in cell_rows if row.get(metric) is not None]
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
    llm_rows = [row for row in rows if row["condition_id"] == LLM_CONDITION]
    agent_steps = config.agent_count * config.steps
    queried = [int(row["agent_steps_queried"]) for row in llm_rows]
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    dispositions = (
        {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        if config.replicates < required
        else decide(rows, contract)
    )
    return {
        "schema_version": "pivot_llm_agent_reference_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "llm_calibration": dict(calibration),
        "query_profile": {
            "agent_steps_per_run": agent_steps,
            "mean_agent_steps_queried_per_run": float(np.mean(queried)) if queried else None,
            "probe_queries_per_run": int(contract["probe_cohort"]["agent_count"]),
        },
        "cells": cells,
        "abm_runs": len(rows),
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot LLM-agent reference scorecard v1",
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
    if "nabm_versus_llm_agent" in dispositions:
        lines.extend(
            [
                "",
                "## Frozen dispositions",
                "",
                f"- NABM versus LLM agent: `{dispositions['nabm_versus_llm_agent']['summary']}`",
                f"- LLM agent versus hand: `{dispositions['llm_agent_versus_hand']['summary']}`",
            ]
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
    timing: Mapping[str, Any],
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
        "schema_version": "pivot_llm_agent_reference_receipt.v1",
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
    # Timing is not byte-reproducible, so it sits outside the receipt.
    delayed._write_exclusive(
        output_dir / "cost_timing.json", json.dumps(timing, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def cost_timing(
    contract: Mapping[str, Any],
    profile: str,
    nabm_seconds: Sequence[float],
    llm_timing: Sequence[Mapping[str, float]],
    reader: Any,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config = abm1.study_config(contract, profile)
    agent_steps = config.agent_count * config.steps
    arms_per_task = len(NABM_ARMS)
    per_arm_cpu = [seconds / arms_per_task for seconds in nabm_seconds]
    llm_gpu = [row["gpu_seconds"] for row in llm_timing]
    queries = [row["forward_passes"] for row in llm_timing]
    floor_spec = contract["cost_floor"]
    floor = json.loads(
        pilot4._bound(floor_spec["scorecard"], floor_spec["sha256"], "the v4 scorecard").read_text(
            encoding="utf-8"
        )
    )["phase_a"]["models"]
    return {
        "status": "TIMING_NOT_BYTE_REPRODUCIBLE",
        "clock": {"nabm": "PROCESS_CPU_TIME_PER_TASK_SINGLE_THREAD_WORKERS", "llm": "GPU_WALL_TIME_PER_FORWARD_PASS"},
        "agent_steps_per_run": agent_steps,
        "nabm_cpu_seconds_per_arm_run": {"median": float(np.median(per_arm_cpu)), "mean": float(np.mean(per_arm_cpu))},
        "llm_agent": {
            "median_gpu_seconds_per_run": float(np.median(llm_gpu)),
            "median_forward_passes_per_run": float(np.median(queries)),
            "total_gpu_seconds": float(np.sum(llm_gpu)),
            "total_forward_passes": int(np.sum(queries)),
            "median_wall_seconds_per_run": float(np.median([row["wall_seconds"] for row in llm_timing])),
            "gpu_seconds_per_forward_pass": float(np.sum(llm_gpu) / max(1, np.sum(queries))),
            "reader_environment": getattr(reader, "environment", None),
            "weight_shard_sha256": getattr(reader, "weight_shard_sha256", None),
        },
        "cost_floor_from_v4_batch_32": {
            model: {
                "seconds_per_query": floor[model]["seconds_per_query"]["batch_32"],
                "gpu_seconds_per_run_at_this_query_count": floor[model]["seconds_per_query"]["batch_32"]
                * float(np.median(queries)),
            }
            for model in floor_spec["models"]
        },
        "llm_rows": len([row for row in rows if row["condition_id"] == LLM_CONDITION]),
        "machine_load_average": list(os.getloadavg()),
    }


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": pilot1._sha256(config_path),
        "conditions": list(CONDITION_IDS),
        "example_prompt": agent_prompt(
            contract, np.full(len(FEATURES), 0.5), contract["agent_prompt"]["no_evidence_text"]
        ),
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
    config = abm1.study_config(contract, args.profile)
    for replicate in range(config.replicates):
        check_llm_machinery_nests_the_fixed_rule(contract, args.profile, replicate, "linear_fit")
    prior = load_v4_prior(contract)
    nabm_rows, nabm_seconds = run_nabm(contract, args.profile, prior, workers=args.workers)
    reader = LLMReader(contract)
    calibration = phase_a(contract, reader)
    if calibration["determinism"]["repeat_identical"] is not True:
        raise RuntimeError("repeated LLM forward passes differ")
    llm_rows, llm_timing = run_llm(contract, args.profile, reader, calibration["calibration_shift"])
    rows = [*nabm_rows, *llm_rows]
    summary = {
        **summarize(rows, contract, args.profile, calibration),
        "precommit_sha": args.precommit_sha,
    }
    timing = cost_timing(contract, args.profile, nabm_seconds, llm_timing, reader, rows)
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
        timing=timing,
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
