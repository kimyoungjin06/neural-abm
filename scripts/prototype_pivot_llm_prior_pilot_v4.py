#!/usr/bin/env python3
"""Run the precommitted Pivot local-LLM prior pilot v4 (model-size ladder).

``elicit`` needs transformers and a GPU; run it once per model through an
ephemeral overlay:

    uv run --with 'transformers==5.14.1' --with accelerate \
        python scripts/prototype_pivot_llm_prior_pilot_v4.py elicit --model qwen3_0_6b ...

``run`` calibrates every model's prior with aggregate information only and
evaluates it in the v3 ABM design, using the project environment only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v4.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_prior_pilot_v2 as pilot2  # noqa: E402
import prototype_pivot_llm_prior_pilot_v3 as pilot3  # noqa: E402


pilot1 = pilot3.pilot1
abm1 = pilot1.abm1
abm2 = pilot1.abm2
delayed = pilot1.delayed
persistence = pilot1.persistence
study1 = pilot1.study1
study2 = pilot1.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = pilot1.FIT_WORLDS
FEATURES = pilot1.FEATURES
ABM_ARMS = pilot3.ABM_ARMS
FIELD_ARMS = pilot3.FIELD_ARMS
SAME_RELEASE_LADDER = ("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b")
ANCHOR = "qwen3_4b_2507"
MODEL_IDS = (*SAME_RELEASE_LADDER, ANCHOR)
REFERENCE = "v3_reference"
PRIOR_SOURCES = (*MODEL_IDS, REFERENCE)
CONDITION_IDS = (
    "hand",
    "flat",
    *(f"{kind}_{source}" for source in PRIOR_SOURCES for kind in ("selective", "hybrid")),
)
ELICITATION_FILES = ("elicitation.jsonl", "llm_prior.json", "elicitation_receipt.json")
OPENED_SEEDS = pilot3.OPENED_SEEDS | {20261023, 20261024}
V1_DESIGN_SEED = 20261016
NONINFERIORITY_MARGIN = pilot3.NONINFERIORITY_MARGIN


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("LLM prior v4 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_llm_prior_pilot.v4":
        raise ValueError("unexpected LLM prior v4 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("LLM prior v4 contract is not a precommit design")
    if [row["id"] for row in contract["models"]] != list(MODEL_IDS):
        raise ValueError("model ladder changed")
    readout = contract["elicitation"]["readout"]
    if not readout["assistant_prefill"].endswith("**"):
        raise ValueError("the registered prefill ends in markdown bold before the answer token")
    if [row["id"] for row in contract["conditions"]] != list(CONDITION_IDS):
        raise ValueError("conditions changed")
    for row in contract["conditions"]:
        expected = 0.0 if row["id"].startswith(("hand", "flat", "selective_")) else float(
            contract["exploration_floor"]["reference_share"]
        )
        if float(row["exploratory_share"]) != expected:
            raise ValueError(f"{row['id']} must use exploratory share {expected}")
    if [tuple(row) for row in contract["abm_arms"]] != list(ABM_ARMS):
        raise ValueError("ABM arms changed")
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if contract["elicitation"]["answer_tokens"] != ["Yes", "No"]:
        raise ValueError("answer tokens must be the strings Yes and No")
    if int(contract["elicitation"]["design_cohort"]["base_seed"]) != V1_DESIGN_SEED:
        raise ValueError("the design cohort must be the v1 cohort so every model sees v1's prompts")
    config = contract["profiles"]["development"]
    open_steps = int(config["steps"]) - int(config["burn_in_steps"])
    if int(contract["exploration_floor"]["open_steps"]) != open_steps:
        raise ValueError("exploration open steps must match the development profile")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != pilot1.IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if tuple(decisions["field_arms"]) != FIELD_ARMS:
        raise ValueError("decision cells changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this pilot")
    seeds = {int(config["base_seed"]), int(contract["probe_cohort"]["base_seed"])}
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("v4 needs fresh seed namespaces")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from ABM v2")
    if contract["boundaries"].get("paid_api_calls") != 0:
        raise ValueError("v4 makes no paid API calls")


def _model(contract: Mapping[str, Any], model_key: str) -> Mapping[str, Any]:
    return next(row for row in contract["models"] if row["id"] == model_key)


def _condition(contract: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    return next(row for row in contract["conditions"] if row["id"] == condition_id)


def _bound(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path)
    if pilot1._sha256(resolved) != expected:
        raise RuntimeError(f"{label} changed")
    return resolved


# -- design cohort ----------------------------------------------------------------


def design_cohort_worlds(contract: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The v1 design cohort's features and its productive labels in each fit world."""

    design = contract["elicitation"]["design_cohort"]
    base_seed = int(design["base_seed"])
    threshold = float(contract["inherited_learning_parameters"]["productive_threshold"])
    scenario = abm1._scenario(contract)
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, list[float]] = defaultdict(list)
    for replicate in range(int(design["replicates"])):
        for world in FIT_WORLDS:
            context = SimpleNamespace(
                scenario=scenario,
                replicate=replicate,
                rng=study1.component_rng(base_seed, replicate, "population"),
            )
            config = study1.StudyConfig(
                agent_count=int(design["agents_per_replicate"]), base_seed=base_seed
            )
            researchers = abm1.apply_fit_world(study1.make_researchers(context, config), world)
            features[world].extend(study2._feature_vector(row) for row in researchers)
            labels[world].extend(float(row.productive_fit >= threshold) for row in researchers)
    stacked = {world: np.stack(rows) for world, rows in features.items()}
    reference = stacked[FIT_WORLDS[0]]
    if any(not np.array_equal(reference, rows) for rows in stacked.values()):
        raise RuntimeError("fit worlds must not change the design-cohort features")
    return reference, {world: np.asarray(values) for world, values in labels.items()}


# -- elicitation (GPU) ------------------------------------------------------------


def render_chat(
    tokenizer: Any, system: str, texts: Sequence[str], template_kwargs: Mapping[str, Any]
) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            **dict(template_kwargs),
        )
        for text in texts
    ]


def load_v1_prompt_hashes(contract: Mapping[str, Any]) -> list[str]:
    spec = contract["v3_reference"]["v1_elicitation"]
    path = _bound(
        Path(spec["directory"]) / "elicitation.jsonl",
        spec["sha256"]["elicitation.jsonl"],
        "the v1 elicitation",
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["primary_prompt_sha256"] for row in rows]


def elicit(contract: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    if model_key not in MODEL_IDS:
        raise ValueError(f"{model_key} is not elicited in v4")
    spec = _model(contract, model_key)
    runtime = contract["runtime"]
    if transformers.__version__ != runtime["transformers"]:
        raise RuntimeError(f"transformers {transformers.__version__} is not the pinned version")
    cached_config = try_to_load_from_cache(spec["model_id"], "config.json", revision=spec["snapshot"])
    if not isinstance(cached_config, str):
        raise RuntimeError("the pinned model snapshot is not in the local cache")
    snapshot_dir = Path(cached_config).parent
    for name, expected in spec["snapshot_file_sha256"].items():
        _bound(snapshot_dir / name, expected, f"model snapshot file {name}")
    shard_hashes = {
        path.name: pilot1._sha256(path) for path in sorted(snapshot_dir.glob("*.safetensors"))
    }
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["snapshot"], local_files_only=True
    )
    tokenizer.padding_side = runtime["padding_side"]
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        revision=spec["snapshot"],
        local_files_only=True,
        dtype=getattr(torch, runtime["dtype"]),
        device_map=runtime["device"],
    )
    model.eval()
    load_seconds = time.time() - load_start
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    yes_id, no_id = pilot1._answer_token_ids(tokenizer, contract["elicitation"]["answer_tokens"])
    system = contract["elicitation"]["system_prompt"]
    features, keys = pilot1.design_cohort(contract)
    prompts = pilot1.render_prompts(contract, features)["primary"]
    prompt_hashes = [pilot1._text_sha256(text) for text in prompts]
    if prompt_hashes != load_v1_prompt_hashes(contract):
        raise RuntimeError("the v4 prompts differ from the v1 prompts")
    template_kwargs = spec["chat_template_kwargs"]
    rendered = render_chat(tokenizer, system, prompts[:1], template_kwargs)[0]
    if not rendered.endswith(spec["generation_prompt_suffix"]):
        raise RuntimeError("the chat template does not end in the registered generation suffix")
    prefill = contract["elicitation"]["readout"]["assistant_prefill"]

    def primary_pass(
        texts: Sequence[str], size: int, suffix: str = prefill
    ) -> tuple[np.ndarray, ...]:
        odds, mass, top_is_answer, seconds = [], [], [], []
        for start in range(0, len(texts), size):
            chats = render_chat(tokenizer, system, texts[start : start + size], template_kwargs)
            batch = tokenizer(
                [text + suffix for text in chats], return_tensors="pt", padding=True
            ).to(runtime["device"])
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
        return (
            np.concatenate(odds),
            np.concatenate(mass),
            np.concatenate(top_is_answer),
            np.asarray(seconds),
        )

    batch_size = int(runtime["batch_size"])
    sensitivity_size = int(runtime["sensitivity_batch_size"])
    primary_odds, answer_mass, top_answer, primary_seconds = primary_pass(prompts, batch_size)
    repeat_odds, *_rest = primary_pass(prompts, batch_size)
    sensitivity_odds, _mass, _top, sensitivity_seconds = primary_pass(
        prompts[:sensitivity_size], sensitivity_size
    )
    _v1_odds, v1_mass, v1_top, _v1_seconds = primary_pass(prompts, batch_size, suffix="")
    probability = 1.0 / (1.0 + np.exp(-primary_odds))
    distilled = pilot1.distill(
        features,
        pilot1.log_odds(probability),
        float(contract["distillation"]["ridge_lambda_per_profile"]),
    )
    fitted_probability = 1.0 / (1.0 + np.exp(-distilled.pop("fitted_log_odds")))
    records = [
        {
            "index": index,
            "replicate": key[0],
            "agent_id": key[1],
            "features": [round(float(value), 10) for value in row],
            "primary_prompt_sha256": prompt_hashes[index],
            "primary_log_odds": float(primary_odds[index]),
            "primary_probability": float(probability[index]),
            "answer_token_mass": float(answer_mass[index]),
            "top_token_is_answer": bool(top_answer[index]),
            "distilled_probability": float(fitted_probability[index]),
        }
        for index, (key, row) in enumerate(zip(keys, features, strict=True))
    ]
    llm_prior = {
        "schema_version": "pivot_llm_prior.v4",
        "model": model_key,
        "features": list(FEATURES),
        **distilled,
        "fidelity": {
            "r_squared_on_log_odds": distilled["r_squared_on_log_odds"],
            "spearman_primary_vs_distilled": float(
                spearmanr(probability, fitted_probability).statistic
            ),
        },
    }
    receipt = {
        "model": model_key,
        "model_id": spec["model_id"],
        "snapshot": spec["snapshot"],
        "weight_shard_sha256": shard_hashes,
        "parameters": parameters,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "profiles": len(records),
        "prompts_identical_to_v1": True,
        "generation_prompt_suffix_verified": True,
        "assistant_prefill": prefill,
        "v1_first_token_readout_validity": {
            "median_answer_token_mass": float(np.median(v1_mass)),
            "share_top_token_is_answer": float(np.mean(v1_top)),
            "weak_format": bool(np.median(v1_mass) < 0.5),
        },
        "prompts_evaluated": {
            "primary": len(records),
            "primary_repeat": len(records),
            "primary_batch_sensitivity": sensitivity_size,
            "v1_readout_validity_diagnostic": len(records),
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
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
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
        "schema_version": "pivot_llm_prior_elicitation_receipt.v4",
        "status": "COMPLETE",
        "contract_sha256": pilot1._sha256(contract_path),
        "precommit": dict(precommit),
        **result["receipt"],
        "outputs": {name: pilot1._sha256(output_dir / name) for name in files},
    }
    delayed._write_exclusive(
        output_dir / "elicitation_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


# -- priors -----------------------------------------------------------------------


def load_elicitation(directory: Path, precommit_sha: str | None) -> dict[str, Any]:
    receipt = json.loads((directory / "elicitation_receipt.json").read_text(encoding="utf-8"))
    if precommit_sha is not None and receipt["precommit"]["commit_sha"] != precommit_sha:
        raise RuntimeError("the elicitation was produced under a different precommit")
    for name, expected in receipt["outputs"].items():
        _bound(directory / name, expected, f"elicitation artifact {name}")
    records = [
        json.loads(line)
        for line in (directory / "elicitation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prior = json.loads((directory / "llm_prior.json").read_text(encoding="utf-8"))
    return {"records": records, "llm_prior": prior, "receipt": receipt}


def load_elicitations(
    contract: Mapping[str, Any], root: Path, precommit_sha: str | None
) -> dict[str, dict[str, Any]]:
    """Every ladder model from this precommit's root; the v1 elicitation by hash."""

    loaded = {model: load_elicitation(root / model, precommit_sha) for model in MODEL_IDS}
    prefill = contract["elicitation"]["readout"]["assistant_prefill"]
    for model in MODEL_IDS:
        receipt = loaded[model]["receipt"]
        if receipt["model"] != model or receipt["prompts_identical_to_v1"] is not True:
            raise RuntimeError(f"{model} elicitation is not the registered model on v1 prompts")
        if receipt["assistant_prefill"] != prefill:
            raise RuntimeError(f"{model} elicitation used a different readout")
    spec = contract["v3_reference"]["v1_elicitation"]
    directory = Path(spec["directory"])
    for name in ELICITATION_FILES:
        _bound(directory / name, spec["sha256"][name], f"v1 elicitation artifact {name}")
    loaded[REFERENCE] = load_elicitation(directory, None)
    return loaded


def calibrate(
    contract: Mapping[str, Any], records: Sequence[Mapping[str, Any]], llm_prior: Mapping[str, Any]
) -> dict[str, Any]:
    calibration = contract["calibration"]
    features = np.asarray([row["features"] for row in records], dtype=np.float64)
    weights = [float(value) for value in llm_prior["weights"]]
    bias = float(llm_prior["bias"])
    target = float(calibration["stated_population_base_rate"])
    shifted = pilot2.calibrate_intercept(
        weights,
        bias,
        features,
        target,
        interval=calibration["bisection_interval"],
        tolerance=float(calibration["tolerance"]),
    )
    return {
        "weights": list(weights),
        "bias": shifted,
        "intercept_shift": shifted - bias,
        "achieved_mean": pilot2.mean_probability(weights, shifted, features),
    }


def build_priors(
    contract: Mapping[str, Any], elicitations: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if study2._initial_policy is not pilot1.HAND_INITIAL_POLICY:
        raise RuntimeError("the prior slot must hold the hand prior while priors are built")
    priors = {
        source: calibrate(
            contract, elicitations[source]["records"], elicitations[source]["llm_prior"]
        )
        for source in PRIOR_SOURCES
    }
    spec = contract["v3_reference"]["v2_calibrated_priors"]
    stored = json.loads(
        _bound(spec["path"], spec["sha256"], "the v2 calibrated priors").read_text(encoding="utf-8")
    )["priors"][spec["prior_used"]]
    reference = priors[REFERENCE]
    if reference["weights"] != [float(v) for v in stored["weights"]] or reference[
        "bias"
    ] != float(stored["bias"]):
        raise RuntimeError("the rebuilt v3 reference prior does not equal the v2 calibrated prior")
    return priors


def phase_a_profile(
    contract: Mapping[str, Any],
    elicitations: Mapping[str, Mapping[str, Any]],
    priors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Readout validity, fidelity, ranking against labels and the anchor, and cost."""

    features, labels = design_cohort_worlds(contract)

    def readout(source: str) -> np.ndarray:
        return np.asarray([row["primary_probability"] for row in elicitations[source]["records"]])

    anchor_probability = readout(ANCHOR)
    v1_probability = readout(REFERENCE)
    hand = abm1._prior_output(features)
    profile: dict[str, Any] = {
        "hand_design_cohort_auc": {
            world: float(roc_auc_score(labels[world], hand)) for world in FIT_WORLDS
        },
        "design_cohort_productive_rate": {
            world: float(np.mean(labels[world])) for world in FIT_WORLDS
        },
        "models": {},
    }
    for model in PRIOR_SOURCES:
        records = elicitations[model]["records"]
        stored = np.asarray([row["features"] for row in records], dtype=np.float64)
        if not np.allclose(stored, features, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"{model} elicitation is not on the design cohort")
        probability = readout(model)
        weights = np.asarray(priors[model]["weights"])
        prior_logit = features @ weights + float(priors[model]["bias"])
        receipt = elicitations[model]["receipt"]
        timing = receipt["timing_seconds"]
        profile["models"][model] = {
            "parameters": receipt.get("parameters"),
            "validity": receipt["validity"],
            "determinism": receipt["determinism"],
            "distillation_fidelity": elicitations[model]["llm_prior"]["fidelity"],
            "raw_mean_probability": float(np.mean(probability)),
            "calibration_intercept_shift": priors[model]["intercept_shift"],
            "readout": "V1_FIRST_TOKEN" if model == REFERENCE else "PREFILLED_FORCED_CHOICE",
            "v1_first_token_readout_validity": receipt.get(
                "v1_first_token_readout_validity", receipt["validity"]
            ),
            "spearman_readout_versus_anchor": float(
                spearmanr(probability, anchor_probability).statistic
            ),
            "spearman_readout_versus_v1_readout": float(
                spearmanr(probability, v1_probability).statistic
            ),
            "calibrated_prior_design_cohort_auc": {
                world: float(roc_auc_score(labels[world], prior_logit)) for world in FIT_WORLDS
            },
            "seconds_per_query": {
                "batch_1": timing["primary_forward_per_prompt"],
                "batch_32": timing["primary_forward_per_prompt_in_sensitivity_batch"],
            },
            "one_time_elicitation_gpu_seconds": timing["primary_forward_total"],
            "peak_gpu_memory_bytes": receipt.get("peak_gpu_memory_bytes"),
        }
    return profile


def cost_ledger(contract: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    config = abm1.study_config(contract, "development")
    agent_steps = len(FIT_WORLDS) * config.replicates * config.agent_count * config.steps
    return {
        "llm_agent_arm_agent_steps": agent_steps,
        "note": "ONE_QUERY_PER_AGENT_STEP_FOR_ONE_LLM_AGENT_ARM_OVER_THE_V4_WORLDS_AND_REPLICATES_A_LOWER_BOUND",
        "by_model": {
            model: {
                "one_time_elicitation_gpu_seconds": entry["one_time_elicitation_gpu_seconds"],
                "llm_agent_arm_gpu_hours": {
                    batch: seconds * agent_steps / 3600.0
                    for batch, seconds in entry["seconds_per_query"].items()
                },
            }
            for model, entry in profile["models"].items()
        },
    }


# -- ABM --------------------------------------------------------------------------


def prior_for(
    condition: Mapping[str, Any], priors: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    if condition["prior"] == "hand":
        return None
    if condition["prior"] == "flat":
        return {"weights": [0.0] * len(FEATURES), "bias": 0.0}
    if condition["prior"] in PRIOR_SOURCES:
        prior = priors[condition["prior"]]
        return {"weights": prior["weights"], "bias": prior["bias"]}
    raise ValueError(f"unknown prior: {condition['prior']}")


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    world_id: str,
    condition_id: str,
    priors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    condition = _condition(contract, condition_id)
    share = float(condition["exploratory_share"])
    epsilon = pilot3.epsilon_for(share, config.steps - config.burn_in_steps)
    prior = prior_for(condition, priors)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows, runs = [], {}
    for arm_id, evidence_regime in ABM_ARMS:
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
    return rows


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_study(
    contract: Mapping[str, Any],
    profile: str,
    priors: Mapping[str, Mapping[str, Any]],
    workers: int = 1,
    conditions: Sequence[str] = CONDITION_IDS,
) -> list[dict[str, Any]]:
    config = abm1.study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, world_id, condition_id, priors)
        for replicate in range(config.replicates)
        for world_id in FIT_WORLDS
        for condition_id in conditions
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


# -- decisions --------------------------------------------------------------------


def _values(indexed, replicates, world, condition, arm, metric) -> np.ndarray:  # noqa: ANN001
    return pilot3._values(indexed, replicates, world, condition, arm, metric)


def _count_label(count: int, total: int, stem: str) -> str:
    if count == total:
        return f"{stem}_IN_ALL_CELLS"
    if count * 2 > total:
        return f"{stem}_IN_MAJORITY"
    if count > 0:
        return f"{stem}_IN_MINORITY"
    return f"{stem}_NOWHERE"


def hybrid_versus_hand(indexed, replicates, contract, *, model, world, arm) -> dict[str, Any]:  # noqa: ANN001
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    hybrid = f"hybrid_{model}"
    share = "productive_share_of_entrants"
    precision = abm1._judge(
        _values(indexed, replicates, world, hybrid, arm, share)
        - _values(indexed, replicates, world, "hand", arm, share),
        float(decision["precision_versus_hand"]["improvement_at_least"]),
        required,
    )
    gap = _values(indexed, replicates, world, "hand", arm, "probe_brier") - _values(
        indexed, replicates, world, hybrid, arm, "probe_brier"
    )
    noninferior = {
        "hand_minus_hybrid_probe_brier": abm1.v2._distribution(gap),
        "pass": float(np.mean(gap)) >= NONINFERIORITY_MARGIN,
    }
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
    }


def selective_versus_hand(indexed, replicates, contract, *, model, world, arm) -> dict[str, Any]:  # noqa: ANN001
    decision = contract["decision_contract"]
    share = "productive_share_of_entrants"
    return abm1._judge(
        _values(indexed, replicates, world, f"selective_{model}", arm, share)
        - _values(indexed, replicates, world, "hand", arm, share),
        float(decision["precision_versus_hand"]["improvement_at_least"]),
        int(decision["improved_replicates_at_least"]),
    )


def hybrid_versus_anchor(indexed, replicates, contract, *, model, world, arm) -> dict[str, Any]:  # noqa: ANN001
    rule = contract["decision_contract"]["hybrid_versus_anchor"]
    share = "productive_share_of_entrants"
    precision_gap = _values(indexed, replicates, world, f"hybrid_{model}", arm, share) - _values(
        indexed, replicates, world, f"hybrid_{ANCHOR}", arm, share
    )
    learning_gap = _values(
        indexed, replicates, world, f"hybrid_{ANCHOR}", arm, "probe_brier"
    ) - _values(indexed, replicates, world, f"hybrid_{model}", arm, "probe_brier")
    precision_ok = float(np.mean(precision_gap)) >= float(rule["precision_margin"])
    learning_ok = float(np.mean(learning_gap)) >= float(rule["learning_margin"])
    if precision_ok and learning_ok:
        disposition = "MATCHES_ANCHOR"
    elif learning_ok:
        disposition = "PRECISION_SHORTFALL"
    elif precision_ok:
        disposition = "LEARNING_SHORTFALL"
    else:
        disposition = "BOTH_SHORTFALL"
    return {
        "disposition": disposition,
        "model_minus_anchor_precision": abm1.v2._distribution(precision_gap),
        "anchor_minus_model_probe_brier": abm1.v2._distribution(learning_gap),
    }


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), row["world_id"], row["condition_id"], row["arm_id"]): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    cells = [(world, arm) for world in FIT_WORLDS for arm in FIELD_ARMS]
    by_model: dict[str, Any] = {}
    for model in PRIOR_SOURCES:
        common = {"model": model}
        hybrid = {
            f"{world}/{arm}": hybrid_versus_hand(
                indexed, replicates, contract, world=world, arm=arm, **common
            )
            for world, arm in cells
        }
        selective = {
            f"{world}/{arm}": selective_versus_hand(
                indexed, replicates, contract, world=world, arm=arm, **common
            )
            for world, arm in cells
        }
        dominated = sorted(key for key, cell in hybrid.items() if cell["disposition"] == "HYBRID_DOMINATES_HAND")
        selects = sorted(key for key, cell in selective.items() if cell["pass"])
        entry: dict[str, Any] = {
            "hybrid_summary": _count_label(len(dominated), len(cells), "DOMINATES"),
            "dominating_cells": dominated,
            "hybrid_cells": hybrid,
            "selective_summary": _count_label(len(selects), len(cells), "SELECTS_BETTER"),
            "selecting_cells": selects,
            "selective_cells": selective,
        }
        if model in SAME_RELEASE_LADDER:
            anchor = {
                f"{world}/{arm}": hybrid_versus_anchor(
                    indexed, replicates, contract, world=world, arm=arm, **common
                )
                for world, arm in cells
            }
            matching = sorted(key for key, cell in anchor.items() if cell["disposition"] == "MATCHES_ANCHOR")
            entry.update(
                {
                    "anchor_summary": _count_label(len(matching), len(cells), "MATCHES_ANCHOR"),
                    "matching_cells": matching,
                    "anchor_cells": anchor,
                }
            )
        by_model[model] = entry
    share = "productive_share_of_entrants"
    readout_sensitivity = {
        f"{world}/{arm}": {
            "anchor_minus_reference_precision": abm1.v2._distribution(
                _values(indexed, replicates, world, f"hybrid_{ANCHOR}", arm, share)
                - _values(indexed, replicates, world, f"hybrid_{REFERENCE}", arm, share)
            ),
            "reference_minus_anchor_probe_brier": abm1.v2._distribution(
                _values(indexed, replicates, world, f"hybrid_{REFERENCE}", arm, "probe_brier")
                - _values(indexed, replicates, world, f"hybrid_{ANCHOR}", arm, "probe_brier")
            ),
        }
        for world, arm in cells
    }
    majority = {"DOMINATES_IN_ALL_CELLS", "DOMINATES_IN_MAJORITY"}
    ladder = [
        model for model in SAME_RELEASE_LADDER if by_model[model]["hybrid_summary"] in majority
    ]
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "by_model": by_model,
        "size_frontier": {
            "status": "DESCRIPTIVE_FRONTIER_NOT_MODEL_SELECTION",
            "same_release_models_dominating_in_majority": ladder,
            "smallest_same_release_model_dominating_in_majority": ladder[0] if ladder else "NONE",
        },
        "readout_sensitivity": {
            "status": "DESCRIPTIVE_SAME_MODEL_PREFILLED_VERSUS_V1_READOUT",
            "cells": readout_sensitivity,
        },
        "v3_replication": (
            "V3_HYBRID_20_MAJORITY_REPLICATED_ON_FRESH_SEEDS"
            if by_model[REFERENCE]["hybrid_summary"] in majority
            else "V3_HYBRID_20_MAJORITY_NOT_REPLICATED"
        ),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
    priors: Mapping[str, Mapping[str, Any]],
    phase_a: Mapping[str, Any],
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
        "schema_version": "pivot_llm_prior_pilot_scorecard.v4",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "calibrated_priors": {model: dict(prior) for model, prior in priors.items()},
        "phase_a": dict(phase_a),
        "cost_ledger": cost_ledger(contract, phase_a),
        "epsilon_by_condition": {
            row["id"]: pilot3.epsilon_for(
                row["exploratory_share"], config.steps - config.burn_in_steps
            )
            for row in contract["conditions"]
        },
        "cells": cells,
        "abm_runs": len(rows),
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot local-LLM prior pilot scorecard v4",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "| Model | Answer mass | Readout vs anchor | Linear AUC | Smooth AUC | Axis AUC | s/query (1) | s/query (32) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, entry in summary["phase_a"]["models"].items():
        auc = entry["calibrated_prior_design_cohort_auc"]
        lines.append(
            f"| {model} | {entry['validity']['median_answer_token_mass']:.3f} | "
            f"{entry['spearman_readout_versus_anchor']:.3f} | {auc['linear_fit']:.3f} | "
            f"{auc['smooth_interaction_fit']:.3f} | {auc['axis_interaction_fit']:.3f} | "
            f"{entry['seconds_per_query']['batch_1']:.4f} | "
            f"{entry['seconds_per_query']['batch_32']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| World | Condition | Arm | Entry | Exploratory | Entrant precision | Probe Brier | Evidence |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
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
    if "by_model" in dispositions:
        lines.extend(["", "## Frozen dispositions", ""])
        for model, entry in dispositions["by_model"].items():
            anchor = entry.get("anchor_summary", "anchor")
            lines.append(
                f"- {model}: hybrid `{entry['hybrid_summary']}`, selective "
                f"`{entry['selective_summary']}`, versus anchor `{anchor}`"
            )
        frontier = dispositions["size_frontier"]
        lines.append(
            f"- size frontier: `{frontier['smallest_same_release_model_dominating_in_majority']}`"
            f" ({frontier['status']})"
        )
        lines.append(f"- v3 replication: `{dispositions['v3_replication']}`")
    lines.extend(["", "Synthetic worlds only.", ""])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
    elicitation_root: Path,
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
        "schema_version": "pivot_llm_prior_pilot_receipt.v4",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_sha256": pilot1._sha256(contract_path),
        "precommit": dict(precommit),
        "elicitations": {
            model: {
                name: pilot1._sha256(elicitation_root / model / name) for name in ELICITATION_FILES
            }
            for model in MODEL_IDS
        },
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
        "models": list(MODEL_IDS),
        "prior_sources": list(PRIOR_SOURCES),
        "conditions": list(CONDITION_IDS),
        "epsilon_by_condition": {
            row["id"]: pilot3.epsilon_for(
                row["exploratory_share"], config.steps - config.burn_in_steps
            )
            for row in contract["conditions"]
        },
        "source_bindings": delayed.verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "elicit", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=MODEL_IDS)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--elicitation-root", type=Path)
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
        if args.model is None:
            parser.error("elicit requires --model")
        result = elicit(contract, args.model)
        receipt = write_elicitation(args.output_dir.resolve(), result, precommit, config_path)
        print(json.dumps({"status": receipt["status"], "model": args.model}))
        return
    if args.elicitation_root is None:
        parser.error("run requires --elicitation-root")
    root = args.elicitation_root.resolve()
    elicitations = load_elicitations(contract, root, args.precommit_sha)
    priors = build_priors(contract, elicitations)
    phase_a = phase_a_profile(contract, elicitations, priors)
    rows = run_study(contract, args.profile, priors, workers=args.workers)
    summary = {
        **summarize(rows, contract, args.profile, priors, phase_a),
        "precommit_sha": args.precommit_sha,
    }
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
        elicitation_root=root,
    )
    print(json.dumps({"status": receipt["status"], "row_count": receipt["row_count"]}))


if __name__ == "__main__":
    main()
