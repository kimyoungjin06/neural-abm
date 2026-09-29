#!/usr/bin/env python3
"""Run the precommitted Pivot local-LLM prior pilot v1.

``elicit`` needs transformers and a GPU; run it through an ephemeral overlay:

    uv run --with 'transformers==5.14.1' --with accelerate \
        python scripts/prototype_pivot_llm_prior_pilot_v1.py elicit ...

``run`` evaluates the priors in the ABM with the project environment only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import platform
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXAMPLES = ROOT / "examples"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v1.yaml"
for import_path in (ROOT, SCRIPTS, EXAMPLES):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import prototype_pivot_nonlinear_policy_abm_v2 as abm2  # noqa: E402


abm1 = abm2.abm1
delayed = abm2.delayed
persistence = abm2.persistence
study1 = abm2.study1
study2 = abm2.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = abm1.FIT_WORLDS
PRIOR_IDS = ("hand", "llm", "oracle_linear", "flat")
ABM_ARMS = (
    ("fixed_rule", "none"),
    ("online_logistic", "neighbor_local"),
    ("linear_batch", "field_public"),
    ("tree_batch", "field_public"),
    ("mlp_batch", "field_public"),
)
FIELD_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
FEATURES = tuple(study2.FEATURES)
HAND_INITIAL_POLICY = study2._initial_policy
OPENED_SEEDS = {
    20261003, 20261004, 20261005, 20261010, 20261011, 20261012, 20261013,
    20261014, 20261015,
}
IMPROVEMENT_RULE = abm1.IMPROVEMENT_RULE
CLIP = 1e-4


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("LLM prior contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_prior_pilot.v1":
        raise ValueError("unexpected LLM prior schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("LLM prior contract is not a precommit design")
    if tuple(contract["elicitation"]["features_in_prompt_order"]) != FEATURES:
        raise ValueError("prompt feature order must match the policy features")
    if [row["id"] for row in contract["priors"]] != list(PRIOR_IDS):
        raise ValueError("prior set changed")
    if [tuple(row) for row in contract["abm_arms"]] != list(ABM_ARMS):
        raise ValueError("ABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if contract["elicitation"]["answer_tokens"] != ["Yes", "No"]:
        raise ValueError("answer tokens must be the strings Yes and No")
    if [row["id"] for row in contract["visibility_regimes"]] != ["immediate"]:
        raise ValueError("the pilot uses immediate visibility only")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from v2")
    oracle = next(row for row in contract["priors"] if row["id"] == "oracle_linear")
    seeds = [
        int(contract["elicitation"]["design_cohort"]["base_seed"]),
        int(oracle["oracle_cohort_base_seed"]),
        int(contract["profiles"]["development"]["base_seed"]),
        int(contract["probe_cohort"]["base_seed"]),
    ]
    if len(set(seeds)) != 4 or set(seeds) & OPENED_SEEDS:
        raise ValueError("the pilot needs four fresh seed namespaces")
    if contract["boundaries"].get("paid_api_calls") != 0:
        raise ValueError("the pilot makes no paid API calls")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this pilot")


def _sha256(path: Path) -> str:
    return abm1._sha256(path)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- design cohort and prompts ------------------------------------------------


def cohort_features(
    contract: Mapping[str, Any], *, base_seed: int, replicates: int, agents: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Population draws only; no network or dynamics are involved."""

    scenario = abm1._scenario(contract)
    rows, keys = [], []
    for replicate in range(replicates):
        context = SimpleNamespace(
            scenario=scenario,
            replicate=replicate,
            rng=study1.component_rng(base_seed, replicate, "population"),
        )
        config = study1.StudyConfig(agent_count=agents, base_seed=base_seed)
        for researcher in study1.make_researchers(context, config):
            rows.append(study2._feature_vector(researcher))
            keys.append((replicate, researcher.agent_id))
    return np.stack(rows), keys


def design_cohort(contract: Mapping[str, Any]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    design = contract["elicitation"]["design_cohort"]
    return cohort_features(
        contract,
        base_seed=int(design["base_seed"]),
        replicates=int(design["replicates"]),
        agents=int(design["agents_per_replicate"]),
    )


def render_prompts(contract: Mapping[str, Any], features: np.ndarray) -> dict[str, list[str]]:
    elicitation = contract["elicitation"]
    template = elicitation["profile_template"]
    profiles = [
        template.format(**{name: float(value) for name, value in zip(FEATURES, row, strict=True)})
        for row in features
    ]
    return {
        "primary": [f"{profile}\n\n{elicitation['primary_question']}" for profile in profiles],
        "secondary": [f"{profile}\n\n{elicitation['secondary_question']}" for profile in profiles],
    }


def parse_verbalized(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return None
    value = float(match.group())
    return value / 100.0 if 0.0 <= value <= 100.0 else None


# -- distillation and priors ----------------------------------------------------


def log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, CLIP, 1.0 - CLIP)
    return np.log(clipped) - np.log1p(-clipped)


def distill(features: np.ndarray, target: np.ndarray, ridge_per_profile: float) -> dict[str, Any]:
    """Ridge fit of log-odds on raw features; the intercept is not penalized."""

    mean_x = features.mean(axis=0)
    mean_y = float(target.mean())
    centered = features - mean_x
    penalty = ridge_per_profile * features.shape[0] * np.eye(features.shape[1])
    weights = np.linalg.solve(centered.T @ centered + penalty, centered.T @ (target - mean_y))
    bias = mean_y - float(mean_x @ weights)
    fitted = features @ weights + bias
    residual = float(np.sum((target - fitted) ** 2))
    total = float(np.sum((target - mean_y) ** 2))
    return {
        "weights": [float(value) for value in weights],
        "bias": float(bias),
        "r_squared_on_log_odds": 1.0 - residual / total if total > 0 else None,
        "fitted_log_odds": fitted,
    }


def oracle_priors(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    oracle = next(row for row in contract["priors"] if row["id"] == "oracle_linear")
    base_seed = int(oracle["oracle_cohort_base_seed"])
    threshold = float(contract["inherited_learning_parameters"]["productive_threshold"])
    scenario = abm1._scenario(contract)
    priors = {}
    for world in FIT_WORLDS:
        rows, labels = [], []
        for replicate in range(int(oracle["oracle_cohort_replicates"])):
            context = SimpleNamespace(
                scenario=scenario,
                replicate=replicate,
                rng=study1.component_rng(base_seed, replicate, "population"),
            )
            config = study1.StudyConfig(agent_count=256, base_seed=base_seed)
            researchers = abm1.apply_fit_world(
                study1.make_researchers(context, config), world
            )
            rows.extend(study2._feature_vector(row) for row in researchers)
            labels.extend(float(row.productive_fit >= threshold) for row in researchers)
        model = LogisticRegression(C=float(oracle["logistic_c"]), max_iter=5000)
        model.fit(np.stack(rows), np.asarray(labels))
        priors[world] = {
            "weights": [float(value) for value in model.coef_[0]],
            "bias": float(model.intercept_[0]),
        }
    return priors


@contextlib.contextmanager
def prior_slot(prior: Mapping[str, Any] | None) -> Iterator[None]:
    """Replace the study-2 initial policy for everything built inside the block."""

    if prior is None:
        yield
        return
    weights = [float(value) for value in prior["weights"]]
    bias = float(prior["bias"])

    def initial_policy() -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.tensor(weights, dtype=torch.float64)
        offset = torch.tensor(bias, dtype=torch.float64)
        return weight.requires_grad_(True), offset.requires_grad_(True)

    study2._initial_policy = initial_policy
    try:
        yield
    finally:
        study2._initial_policy = HAND_INITIAL_POLICY


def prior_for(
    prior_id: str,
    world_id: str,
    llm_prior: Mapping[str, Any],
    oracles: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if prior_id == "hand":
        return None
    if prior_id == "llm":
        return {"weights": llm_prior["weights"], "bias": llm_prior["bias"]}
    if prior_id == "oracle_linear":
        return oracles[world_id]
    if prior_id == "flat":
        return {"weights": [0.0] * len(FEATURES), "bias": 0.0}
    raise ValueError(f"unknown prior: {prior_id}")


# -- elicitation (GPU) ---------------------------------------------------------


def _answer_token_ids(tokenizer: Any, answers: Sequence[str]) -> list[int]:
    ids = []
    for answer in answers:
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"answer {answer!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def elicit(contract: Mapping[str, Any]) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    llm = contract["llm"]
    runtime = llm["runtime"]
    if transformers.__version__ != runtime["transformers"]:
        raise RuntimeError(f"transformers {transformers.__version__} is not pinned version")
    tokenizer = AutoTokenizer.from_pretrained(
        llm["model_id"], revision=llm["snapshot"], local_files_only=True
    )
    tokenizer.padding_side = runtime["padding_side"]
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        llm["model_id"],
        revision=llm["snapshot"],
        local_files_only=True,
        dtype=getattr(torch, runtime["dtype"]),
        device_map=runtime["device"],
    )
    model.eval()
    load_seconds = time.time() - load_start
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

    cached_config = try_to_load_from_cache(
        llm["model_id"], "config.json", revision=llm["snapshot"]
    )
    if not isinstance(cached_config, str):
        raise RuntimeError("the pinned model snapshot is not in the local cache")
    snapshot_dir = Path(cached_config).parent
    for name, expected in llm["snapshot_file_sha256"].items():
        if _sha256(snapshot_dir / name) != expected:
            raise RuntimeError(f"model snapshot file {name} changed")
    shard_hashes = {
        path.name: _sha256(path) for path in sorted(snapshot_dir.glob("*.safetensors"))
    }
    yes_id, no_id = _answer_token_ids(tokenizer, contract["elicitation"]["answer_tokens"])
    system = contract["elicitation"]["system_prompt"]
    features, keys = design_cohort(contract)
    prompts = render_prompts(contract, features)
    batch_size = int(runtime["batch_size"])
    sensitivity_size = int(runtime["sensitivity_batch_size"])

    def chat(texts: Sequence[str]) -> list[str]:
        return [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in texts
        ]

    def primary_pass(texts: Sequence[str], size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        odds, mass, top_is_answer, seconds = [], [], [], []
        for start in range(0, len(texts), size):
            batch = tokenizer(chat(texts[start : start + size]), return_tensors="pt", padding=True)
            batch = batch.to(runtime["device"])
            torch.cuda.synchronize()
            began = time.time()
            with torch.no_grad():
                logits = model(**batch).logits[:, -1, :].float()
            torch.cuda.synchronize()
            seconds.append(time.time() - began)
            probabilities = torch.softmax(logits, dim=-1)
            odds.append((logits[:, yes_id] - logits[:, no_id]).cpu().numpy())
            mass.append((probabilities[:, yes_id] + probabilities[:, no_id]).cpu().numpy())
            top = logits.argmax(dim=-1)
            top_is_answer.append(((top == yes_id) | (top == no_id)).cpu().numpy())
        return np.concatenate(odds), np.concatenate(mass), np.concatenate(top_is_answer), seconds

    primary_odds, answer_mass, top_answer, primary_seconds = primary_pass(
        prompts["primary"], batch_size
    )
    repeat_odds, _mass, _top, _seconds = primary_pass(prompts["primary"], batch_size)
    sensitivity_odds, _mass, _top, sensitivity_seconds = primary_pass(
        prompts["primary"][:sensitivity_size], sensitivity_size
    )
    verbalized_raw = []
    secondary_seconds = []
    for start in range(0, len(prompts["secondary"]), batch_size):
        batch = tokenizer(
            chat(prompts["secondary"][start : start + batch_size]),
            return_tensors="pt",
            padding=True,
        ).to(runtime["device"])
        torch.cuda.synchronize()
        began = time.time()
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        secondary_seconds.append(time.time() - began)
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        verbalized_raw.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    verbalized = [parse_verbalized(text) for text in verbalized_raw]
    probability = 1.0 / (1.0 + np.exp(-primary_odds))
    distilled = distill(
        features,
        log_odds(probability),
        float(contract["distillation"]["ridge_lambda_per_profile"]),
    )
    fitted_probability = 1.0 / (1.0 + np.exp(-distilled.pop("fitted_log_odds")))
    valid_verbalized = [
        (p, v) for p, v in zip(probability, verbalized, strict=True) if v is not None
    ]
    records = [
        {
            "index": index,
            "replicate": key[0],
            "agent_id": key[1],
            "features": [round(float(value), 10) for value in row],
            "primary_prompt_sha256": _text_sha256(prompts["primary"][index]),
            "primary_log_odds": float(primary_odds[index]),
            "primary_probability": float(probability[index]),
            "answer_token_mass": float(answer_mass[index]),
            "top_token_is_answer": bool(top_answer[index]),
            "distilled_probability": float(fitted_probability[index]),
            "verbalized_raw": verbalized_raw[index],
            "verbalized_probability": verbalized[index],
        }
        for index, (key, row) in enumerate(zip(keys, features, strict=True))
    ]
    llm_prior = {
        "schema_version": "pivot_llm_prior.v1",
        "features": list(FEATURES),
        **distilled,
        "fidelity": {
            "r_squared_on_log_odds": distilled["r_squared_on_log_odds"],
            "spearman_primary_vs_distilled": float(
                spearmanr(probability, fitted_probability).statistic
            ),
            "spearman_primary_vs_verbalized": (
                float(spearmanr(*zip(*valid_verbalized)).statistic)
                if len(valid_verbalized) > 2
                else None
            ),
            "verbalized_parse_rate": len(valid_verbalized) / len(verbalized),
        },
    }
    receipt = {
        "model_id": llm["model_id"],
        "snapshot": llm["snapshot"],
        "weight_shard_sha256": shard_hashes,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "profiles": len(records),
        "prompts_evaluated": {
            "primary": len(records),
            "primary_repeat": len(records),
            "primary_batch_sensitivity": sensitivity_size,
            "secondary_generation": len(records),
        },
        "validity": {
            "median_answer_token_mass": float(np.median(answer_mass)),
            "share_top_token_is_answer": float(np.mean(top_answer)),
            "weak_format": bool(np.median(answer_mass) < 0.5),
        },
        "determinism": {
            "repeat_pass_identical": bool(np.array_equal(primary_odds, repeat_odds)),
            "max_abs_log_odds_canonical_vs_sensitivity_batch": float(
                np.max(np.abs(primary_odds[:sensitivity_size] - sensitivity_odds))
            ),
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "primary_forward_total": float(np.sum(primary_seconds)),
            "primary_forward_per_prompt": float(np.sum(primary_seconds)) / len(records),
            "primary_forward_per_prompt_in_sensitivity_batch": float(
                np.sum(sensitivity_seconds)
            )
            / sensitivity_size,
            "secondary_generation_total": float(np.sum(secondary_seconds)),
        },
    }
    return {"records": records, "llm_prior": llm_prior, "receipt": receipt}


def write_elicitation(
    output_dir: Path, result: Mapping[str, Any], precommit: Mapping[str, Any], contract_path: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "elicitation.jsonl": "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in result["records"]
        ),
        "llm_prior.json": json.dumps(result["llm_prior"], indent=2, sort_keys=True) + "\n",
    }
    for name, content in files.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_llm_prior_elicitation_receipt.v1",
        "status": "COMPLETE",
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        **result["receipt"],
        "outputs": {name: _sha256(output_dir / name) for name in files},
    }
    delayed._write_exclusive(
        output_dir / "elicitation_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


# -- ABM evaluation (CPU) ------------------------------------------------------


def study_config(contract: Mapping[str, Any], profile: str) -> Any:
    return abm1.study_config(contract, profile)


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    world_id: str,
    prior_id: str,
    prior: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    config = study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows, runs = [], {}
    with prior_slot(prior):
        for arm_id, evidence_regime in ABM_ARMS:
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
            runs[arm_id] = run
            rows.append(
                {
                    "replicate": replicate,
                    "world_id": world_id,
                    "prior_id": prior_id,
                    "arm_id": arm_id,
                    "evidence_regime": evidence_regime,
                    **run,
                }
            )
    if study2._initial_policy is not HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot was not restored")
    if prior_id == "hand" and world_id == "linear_fit":
        abm1.check_nesting(
            contract, config, replicate=replicate, visibility_id="immediate", runs=runs
        )
    return rows


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_abm(
    contract: Mapping[str, Any],
    profile: str,
    llm_prior: Mapping[str, Any],
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = study_config(contract, profile)
    oracles = oracle_priors(contract)
    tasks = [
        (
            contract,
            profile,
            replicate,
            world_id,
            prior_id,
            prior_for(prior_id, world_id, llm_prior, oracles),
        )
        for replicate in range(config.replicates)
        for world_id in FIT_WORLDS
        for prior_id in PRIOR_IDS
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
    return [row for rows in results for row in rows], oracles


def _judge_pair(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    *,
    world_id: str,
    arm_id: str,
    metric: str,
    higher_is_better: bool,
    minimum: float,
    required: int,
) -> dict[str, Any]:
    llm = np.asarray([float(indexed[(r, world_id, "llm", arm_id)][metric]) for r in replicates])
    hand = np.asarray([float(indexed[(r, world_id, "hand", arm_id)][metric]) for r in replicates])
    gain = llm - hand if higher_is_better else hand - llm
    return {
        "llm_over_hand": abm1._judge(gain, minimum, required),
        "hand_over_llm": abm1._judge(-gain, minimum, required),
    }


def _three_way(report: Mapping[str, Any], names: Sequence[str]) -> str:
    if report["llm_over_hand"]["pass"]:
        return names[0]
    if report["hand_over_llm"]["pass"]:
        return names[1]
    return names[2]


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), row["world_id"], row["prior_id"], row["arm_id"]): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    by_world = {}
    for world_id in FIT_WORLDS:
        common = {"world_id": world_id, "required": required}
        quality = _judge_pair(
            indexed, replicates, arm_id="fixed_rule", metric="probe_brier",
            higher_is_better=False,
            minimum=float(decision["prior_quality"]["improvement_at_least"]), **common,
        )
        local = _judge_pair(
            indexed, replicates, arm_id="online_logistic", metric="probe_brier",
            higher_is_better=False,
            minimum=float(decision["local_rescue"]["improvement_at_least"]), **common,
        )
        pooled = {
            arm: _judge_pair(
                indexed, replicates, arm_id=arm, metric="probe_brier",
                higher_is_better=False,
                minimum=float(decision["pooled_washout"]["improvement_at_least"]), **common,
            )
            for arm in FIELD_ARMS
        }
        entry = _judge_pair(
            indexed, replicates, arm_id="fixed_rule", metric="productive_share_of_entrants",
            higher_is_better=True,
            minimum=float(decision["entry_precision"]["improvement_at_least"]), **common,
        )
        persists = sorted(
            arm
            for arm, report in pooled.items()
            if report["llm_over_hand"]["pass"] or report["hand_over_llm"]["pass"]
        )
        by_world[world_id] = {
            "prior_quality": _three_way(quality, decision["prior_quality"]["dispositions"]),
            "local_rescue": _three_way(local, decision["local_rescue"]["dispositions"]),
            "pooled_washout": (
                "PRIOR_PERSISTS_UNDER_POOLING" if persists else "PRIOR_WASHES_OUT_UNDER_POOLING"
            ),
            "prior_persists_for_arms": persists,
            "entry_precision": _three_way(entry, decision["entry_precision"]["dispositions"]),
            "reports": {
                "prior_quality": quality,
                "local_rescue": local,
                "pooled_washout": pooled,
                "entry_precision": entry,
            },
        }
    return {"status": "FROZEN_SYNTHETIC_DECISION_COMPLETE", "by_world": by_world}


def summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
    oracles: Mapping[str, Any],
    llm_prior: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = (
        "probe_brier",
        "probe_auc",
        "productive_share_of_entrants",
        "cumulative_entry_rate",
        "active_failed_rate",
    )
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world_id"], row["prior_id"], row["arm_id"])].append(row)
    cells = []
    for world_id in FIT_WORLDS:
        for prior_id in PRIOR_IDS:
            for arm_id, evidence_regime in ABM_ARMS:
                cell_rows = grouped[(world_id, prior_id, arm_id)]
                means = {}
                for metric in metrics:
                    values = [row[metric] for row in cell_rows if row[metric] is not None]
                    means[metric] = round(float(np.mean(values)), 6) if values else None
                cells.append(
                    {
                        "world_id": world_id,
                        "prior_id": prior_id,
                        "arm_id": arm_id,
                        "evidence_regime": evidence_regime,
                        "replicates": len(cell_rows),
                        "means": means,
                    }
                )
    config = study_config(contract, profile)
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    dispositions = (
        {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        if config.replicates < required
        else decide(rows, contract)
    )
    return {
        "schema_version": "pivot_llm_prior_pilot_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "priors": {
            "llm": {"weights": llm_prior["weights"], "bias": llm_prior["bias"]},
            "oracle_linear": dict(oracles),
            "hand": {
                "weights": [float(v) for v in HAND_INITIAL_POLICY()[0].detach()],
                "bias": float(HAND_INITIAL_POLICY()[1].detach()),
            },
        },
        "cells": cells,
        "abm_runs": len(rows),
        "abm_policy_queries": len(rows) * config.agent_count * config.steps,
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot local-LLM prior pilot scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| World | Prior | Arm | Probe Brier | Entrant precision | Entry |",
        "|---|---|---|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        means = cell["means"]
        lines.append(
            f"| {cell['world_id']} | {cell['prior_id']} | {cell['arm_id']} | "
            f"{means['probe_brier']:.4f} | {means['productive_share_of_entrants']:.4f} | "
            f"{means['cumulative_entry_rate']:.4f} |"
        )
    dispositions = summary["dispositions"]
    if "by_world" in dispositions:
        lines.extend(["", "## Frozen dispositions", ""])
        for world_id, entry in dispositions["by_world"].items():
            lines.append(
                f"- {world_id}: prior `{entry['prior_quality']}`, local "
                f"`{entry['local_rescue']}`, pooled `{entry['pooled_washout']}`, entry "
                f"`{entry['entry_precision']}`"
            )
    lines.extend(
        [
            "",
            "Synthetic worlds only. This does not establish that the LLM knows how real "
            "pivots succeed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_abm(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
    elicitation_dir: Path,
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
        "schema_version": "pivot_llm_prior_pilot_receipt.v1",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        "elicitation": {
            name: _sha256(elicitation_dir / name)
            for name in ("elicitation.jsonl", "llm_prior.json", "elicitation_receipt.json")
        },
        "outputs": {name: _sha256(output_dir / name) for name in files},
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(
        output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def load_elicitation(elicitation_dir: Path, precommit_sha: str) -> dict[str, Any]:
    receipt = json.loads((elicitation_dir / "elicitation_receipt.json").read_text(encoding="utf-8"))
    if receipt["precommit"]["commit_sha"] != precommit_sha:
        raise RuntimeError("the elicitation was produced under a different precommit")
    for name, expected in receipt["outputs"].items():
        if _sha256(elicitation_dir / name) != expected:
            raise RuntimeError(f"elicitation artifact {name} changed")
    return json.loads((elicitation_dir / "llm_prior.json").read_text(encoding="utf-8"))


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    features, _keys = design_cohort(contract)
    prompts = render_prompts(contract, features)
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_sha256": _sha256(config_path),
        "profiles": len(features),
        "first_primary_prompt": prompts["primary"][0],
        "priors": list(PRIOR_IDS),
        "abm_arms": [list(arm) for arm in ABM_ARMS],
        "source_bindings": delayed.verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "elicit", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--elicitation-dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error(f"{args.command} requires a full 40-character lowercase --precommit-sha")
    if args.output_dir is None:
        parser.error(f"{args.command} requires --output-dir")
    precommit = delayed.verify_precommit(args.precommit_sha, contract)
    if args.command == "elicit":
        receipt = write_elicitation(args.output_dir.resolve(), elicit(contract), precommit, config_path)
        print(json.dumps({"status": receipt["status"], "profiles": receipt["profiles"]}))
        return
    if args.elicitation_dir is None:
        parser.error("run requires --elicitation-dir")
    llm_prior = load_elicitation(args.elicitation_dir.resolve(), args.precommit_sha)
    rows, oracles = run_abm(contract, args.profile, llm_prior, workers=args.workers)
    summary = {
        **summarize(rows, contract, args.profile, oracles, llm_prior),
        "precommit_sha": args.precommit_sha,
    }
    receipt = write_abm(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
        elicitation_dir=args.elicitation_dir.resolve(),
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
