"""Information-spread policy ladder, study v1 amendment D3b: the LLM arms (stage D5).

Design: docs/research/info-spread-policy-ladder-llm-arms-v1.md
Contract: experiments/configs/info_spread_policy_ladder_llm_arms_v1.yaml

Two commands.

- `elicit` reads the pinned Qwen once, on a GPU, through a transformers
  overlay. It answers every P0 profile and every cell of the X0 table at batch
  1 and writes elicitation.jsonl with a receipt. It reads no world.
- `run` builds P0, X0, and L1 from that elicitation and runs them in the D4
  v2 cells, in the project environment. The D4 v2 arms are read from its
  cells.jsonl, bound by its record, which must come from the same precommit;
  replicate 0 of every L1 task also recomputes L0 and must equal D4 v2's value.

Every arm is scored as in study v2: the primary measure is share precision on
unverified items (info_spread_unverified_measure_v1), and share precision
over all items is recorded beside it.

X0's public aggregates (fact-checked items by class) depend on the item stream
alone, never on decisions. So the table over (archetype, item class,
aggregate) answers every query X0 could make, and computing it before the run
gives the same answers as querying during the run with a prompt cache.

This amendment is frozen in the same commit as study v2, before D4 v2 runs.
A development run needs that precommit SHA, so D5 runs in a checkout of the
freeze commit; smoke and pilot runs skip the check and are never read for
results. cells.jsonl and summary.json carry no timing.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS needs this before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Mapping, Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import info_spread_unverified_measure_v1 as measure  # noqa: E402
import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as d4  # noqa: E402


DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_llm_arms_v1.yaml"
ARMS = ("P0", "X0", "L1")
COMPARISONS = {
    "llm_prior_in_learner": ("L1", "L0"),
    "nabm_vs_archetypes": ("L0", "X0"),
    "closed_loop_on_llm_prior": ("L1", "P0"),
}
DESCRIPTIVE = {"llm_prior_vs_hand": ("P0", "F1"), "archetypes_vs_hand": ("X0", "F1")}
LABELS = ("POSITIVE", "NEGATIVE", "INDISTINGUISHABLE")
PROMPT_SETS = ("P0/A", "P0/B", "X0/A", "X0/B")
CLIP = 1e-4


# -- contract and precommit --------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def load_d4_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return d4.load_contract(ROOT / contract["d4"]["contract"])


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract["status"] != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("not a precommit contract")
    if tuple(contract["arms"]) != ARMS:
        raise ValueError("arms differ from the runner's")
    if {k: tuple(v) for k, v in contract["comparisons"].items()} != COMPARISONS:
        raise ValueError("comparisons differ from the runner's")
    if contract["llm"]["batch_size"] != 1:
        raise ValueError("the canonical elicitation is at batch 1")


def verify_source_bindings(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    return d4.verify_source_bindings(contract)


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    return d4.verify_precommit(commit_sha, contract)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- P0 profiles -------------------------------------------------------------------


def p0_cohort(contract: Mapping[str, Any], regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Design-cohort features in the world's column order, and contact counts.

    Item features and trust are rounded to the two decimals the prompt shows,
    so the distilled prior is fitted on what the model read.
    """

    spec = contract["p0"]["design_cohort"]
    degree = int(spec["degree"])
    rng = np.random.default_rng(core.keyed_seed(int(spec["seed"]), ord(regime)))
    count = int(spec["profiles"])
    items = np.round(rng.random((count, 4)), 2)
    contacts = rng.integers(1, degree + 1, count)
    columns = [items, (contacts / degree)[:, None], np.log1p(contacts)[:, None]]
    if regime == "B":
        columns.append(np.round(rng.random((count, 1)), 2))
    return np.hstack(columns), contacts


def p0_prompts(contract: Mapping[str, Any], regime: str) -> list[str]:
    prompts = contract["prompts"]
    degree = int(contract["p0"]["design_cohort"]["degree"])
    features, contacts = p0_cohort(contract, regime)
    texts = []
    for row, count in zip(features, contacts, strict=True):
        lines = prompts["p0_item"].format(
            credibility=row[0], novelty=row[1], intensity=row[2], fit=row[3], count=int(count), degree=degree,
        )
        if regime == "B":
            lines += "\n" + prompts["p0_trust"].format(trust=row[6])
        texts.append(f"{prompts['p0_lead']}\n\n{lines}\n\n{prompts['question']}")
    return texts


# -- X0 table ----------------------------------------------------------------------


def aggregate_state(n: int, k: int, spec: Mapping[str, Any]) -> tuple[int, int, int]:
    """What the prompt shows of `k` true among `n` fact-checked items.

    Exact counts up to `exact_up_to`; above, the count rounded to
    `count_step` and the percent true rounded to `percent_step`, half up.
    """

    exact, step, percent = int(spec["exact_up_to"]), int(spec["count_step"]), int(spec["percent_step"])
    if n == 0:
        return (0, 0, 0)
    if n <= exact:
        return (1, n, k)
    shown = step * ((2 * n + step) // (2 * step))
    pct = percent * ((200 * k + percent * n) // (2 * percent * n))
    return (2, shown, pct)


def aggregate_states(spec: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    states = {aggregate_state(n, k, spec) for n in range(int(spec["n_max"]) + 1) for k in range(n + 1)}
    return sorted(states)


def archetype_count(contract: Mapping[str, Any], regime: str) -> int:
    x0 = contract["x0"]
    return int(x0["threshold_bins"]) * (len(contract["prompts"]["x0_cost"]) if regime == "B" else 1)


def x0_keys(contract: Mapping[str, Any], regime: str) -> list[tuple[int, int, tuple[int, int, int]]]:
    states = aggregate_states(contract["x0"]["aggregates"])
    classes = int(contract["x0"]["class_bins"]) ** int(contract["x0"]["class_features"])
    return [(a, c, s) for a in range(archetype_count(contract, regime)) for c in range(classes) for s in states]


def _aggregate_text(prompts: Mapping[str, Any], state: tuple[int, int, int]) -> str:
    kind, n, value = state
    if kind == 0:
        return prompts["x0_none"]
    if kind == 1:
        return prompts["x0_exact"].format(n=n, k=value)
    return prompts["x0_rounded"].format(n=n, pct=value)


def x0_prompts(contract: Mapping[str, Any], regime: str) -> list[str]:
    prompts = contract["prompts"]
    costs = len(prompts["x0_cost"])
    texts = []
    for archetype, item_class, state in x0_keys(contract, regime):
        # spread.archetypes numbers Regime B groups as threshold bin * costs + cost index.
        threshold_bin, cost_index = (archetype // costs, archetype % costs) if regime == "B" else (archetype, None)
        person = prompts["x0_threshold"][threshold_bin]
        if cost_index is not None:
            person += " " + prompts["x0_cost"][cost_index]
        # spread.item_classes numbers classes as credibility bin + 2 * novelty bin.
        item = prompts["x0_class"].format(
            credibility=prompts["x0_level"][item_class % 2], novelty=prompts["x0_level"][item_class // 2],
        )
        body = prompts["x0_body"].format(person=person, item=item, aggregate=_aggregate_text(prompts, state))
        texts.append(f"{prompts['x0_lead']}\n\n{body}\n\n{prompts['question']}")
    return texts


def prompt_sets(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        "P0/A": p0_prompts(contract, "A"), "P0/B": p0_prompts(contract, "B"),
        "X0/A": x0_prompts(contract, "A"), "X0/B": x0_prompts(contract, "B"),
    }


# -- elicitation (GPU, transformers overlay) -----------------------------------------


def _answer_token_ids(tokenizer: Any, answers: Sequence[str]) -> list[int]:
    ids = []
    for answer in answers:
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"answer {answer!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def elicit(contract: Mapping[str, Any], limit: int | None = None) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    spec = contract["llm"]
    if transformers.__version__ != spec["transformers"]:
        raise RuntimeError(f"transformers {transformers.__version__} is not the pinned version")
    cached = try_to_load_from_cache(spec["model_id"], "config.json", revision=spec["snapshot"])
    if not isinstance(cached, str):
        raise RuntimeError("the pinned model snapshot is not in the local cache")
    snapshot_dir = Path(cached).parent
    for name, expected in spec["snapshot_file_sha256"].items():
        if _sha256(snapshot_dir / name) != expected:
            raise RuntimeError(f"model snapshot file {name} differs from the pinned hash")
    shards = {path.name: _sha256(path) for path in sorted(snapshot_dir.glob("*.safetensors"))}
    if shards != dict(spec["weight_shard_sha256"]):
        raise RuntimeError("the weight shards differ from the pinned hashes")
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["snapshot"], local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    began = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["snapshot"], local_files_only=True, dtype=getattr(torch, spec["dtype"]),
    ).to("cuda")
    model.eval()
    load_seconds = time.time() - began
    yes_id, no_id = _answer_token_ids(tokenizer, spec["answer_tokens"])
    system = contract["prompts"]["system"]
    prefill = spec["assistant_prefill"]

    def chat(text: str) -> str:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
        if not rendered.endswith(spec["generation_prompt_suffix"]):
            raise RuntimeError("the chat template does not end in the registered generation suffix")
        return rendered + prefill

    def readout(text: str) -> tuple[float, float, bool]:
        batch = tokenizer([chat(text)], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**batch).logits[0, -1, :].float()
        probabilities = torch.softmax(logits, dim=-1)
        top = int(logits.argmax())
        return (float(logits[yes_id] - logits[no_id]), float(probabilities[yes_id] + probabilities[no_id]),
                top in (yes_id, no_id))

    records, seconds = [], {}
    for name, texts in prompt_sets(contract).items():
        texts = texts[:limit] if limit else texts
        torch.cuda.synchronize()
        began = time.time()
        for index, text in enumerate(texts):
            log_odds, mass, top_is_answer = readout(text)
            records.append({
                "set": name, "index": index, "prompt_sha256": _text_sha256(text),
                "log_odds": log_odds, "answer_token_mass": mass, "top_token_is_answer": top_is_answer,
            })
        torch.cuda.synchronize()
        seconds[name] = time.time() - began
    repeat = int(spec["repeat_check_prompts"])
    first = [r for r in records if r["set"] == "X0/A"][:repeat]
    x0_texts = prompt_sets(contract)["X0/A"]
    repeat_identical = all(readout(x0_texts[r["index"]])[0] == r["log_odds"] for r in first)
    masses = np.asarray([r["answer_token_mass"] for r in records])
    receipt = {
        "model_id": spec["model_id"], "snapshot": spec["snapshot"], "weight_shard_sha256": shards,
        "transformers": transformers.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), "node": platform.node(),
        "assistant_prefill": prefill, "batch_size": 1,
        "prompts": {name: sum(1 for r in records if r["set"] == name) for name in PROMPT_SETS},
        "validity": {
            "median_answer_token_mass": float(np.median(masses)),
            "share_top_token_is_answer": float(np.mean([r["top_token_is_answer"] for r in records])),
            "weak_format": bool(np.median(masses) < 0.5),
        },
        "determinism": {"repeat_prompts": len(first), "repeat_identical": bool(repeat_identical)},
        "timing_seconds": {"model_load": load_seconds, **{f"forward/{k}": v for k, v in seconds.items()}},
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return {"records": records, "receipt": receipt}


def write_elicitation(output_dir: Path, result: Mapping[str, Any], contract: Mapping[str, Any],
                      contract_path: Path, precommit: Mapping[str, Any] | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "elicitation.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in result["records"]), encoding="utf-8")
    receipt = {
        "schema_version": "info_spread_llm_arms_elicitation_receipt.v1",
        "status": "REGISTERED" if precommit else "NOT_FOR_RESULTS_SMOKE",
        "contract_sha256": _sha256(contract_path), "precommit": precommit,
        **result["receipt"], "outputs": {"elicitation.jsonl": _sha256(path)},
    }
    (output_dir / "elicitation_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    return receipt


def load_elicitation(directory: Path, contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Log-odds per prompt set, checked against this contract's prompts."""

    receipt = json.loads((directory / "elicitation_receipt.json").read_text(encoding="utf-8"))
    path = directory / "elicitation.jsonl"
    if _sha256(path) != receipt["outputs"]["elicitation.jsonl"]:
        raise RuntimeError("elicitation.jsonl differs from its receipt")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out = {}
    for name, texts in prompt_sets(contract).items():
        chosen = [r for r in rows if r["set"] == name]
        if [r["prompt_sha256"] for r in chosen] != [_text_sha256(t) for t in texts]:
            raise RuntimeError(f"the {name} elicitation does not answer this contract's prompts")
        out[name] = np.asarray([r["log_odds"] for r in chosen], dtype=np.float64)
    return out


# -- P0, X0, and L1 from the elicitation ----------------------------------------------


def _log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, CLIP, 1.0 - CLIP)
    return np.log(clipped) - np.log1p(-clipped)


def distill(features: np.ndarray, target: np.ndarray, ridge_per_profile: float) -> tuple[np.ndarray, float, float]:
    """Ridge fit of log-odds on the features; the intercept is not penalized."""

    mean_x, mean_y = features.mean(axis=0), float(target.mean())
    centered = features - mean_x
    penalty = ridge_per_profile * features.shape[0] * np.eye(features.shape[1])
    weights = np.linalg.solve(centered.T @ centered + penalty, centered.T @ (target - mean_y))
    bias = mean_y - float(mean_x @ weights)
    fitted = features @ weights + bias
    total = float(np.sum((target - mean_y) ** 2))
    r_squared = 1.0 - float(np.sum((target - fitted) ** 2)) / total if total > 0 else float("nan")
    return weights, bias, r_squared


def shift_to_base_rate(logits: np.ndarray, target: float, spec: Mapping[str, Any]) -> float:
    """The single log-odds shift that makes the mean probability equal `target`."""

    low, high = (float(v) for v in spec["bisection_interval"])
    if not core.sigmoid(logits + low).mean() < target < core.sigmoid(logits + high).mean():
        raise RuntimeError("the calibration target is outside the bisection interval")
    for _ in range(400):
        middle = 0.5 * (low + high)
        value = float(core.sigmoid(logits + middle).mean())
        if abs(value - target) < float(spec["tolerance"]):
            return middle
        low, high = (middle, high) if value < target else (low, middle)
    raise RuntimeError("the calibration did not converge")


def build_llm_arms(contract: Mapping[str, Any], log_odds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """P0's calibrated linear priors and X0's shifted tables, per regime."""

    calibration = contract["calibration"]
    target = float(calibration["base_rate"])
    arms: dict[str, Any] = {"P0": {}, "X0": {}, "report": {}}
    for regime in ("A", "B"):
        features, _contacts = p0_cohort(contract, regime)
        target_log_odds = _log_odds(core.sigmoid(log_odds[f"P0/{regime}"]))
        weights, bias, r_squared = distill(features, target_log_odds, float(contract["p0"]["ridge_per_profile"]))
        shift = shift_to_base_rate(features @ weights + bias, target, calibration)
        arms["P0"][regime] = core.LinearPrior(weights, bias + shift)

        states = aggregate_states(contract["x0"]["aggregates"])
        classes = int(contract["x0"]["class_bins"]) ** int(contract["x0"]["class_features"])
        table = log_odds[f"X0/{regime}"].reshape(archetype_count(contract, regime), classes, len(states))
        x_shift = shift_to_base_rate(table[:, :, states.index((0, 0, 0))].ravel(), target, calibration)
        arms["X0"][regime] = {"table": table + x_shift, "states": {s: i for i, s in enumerate(states)}}
        arms["report"][regime] = {
            "p0_weights": [round(float(w), 6) for w in weights], "p0_bias": round(bias, 6),
            "p0_calibration_shift": round(shift, 6), "p0_distillation_r_squared": round(r_squared, 6),
            "x0_calibration_shift": round(x_shift, 6),
        }
    return arms


def class_aggregates(world: spread.World, stream: spread.Stream, classify: Callable[[np.ndarray], np.ndarray],
                     classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Fact-checked and true counts per step and item class, visible at the step's decisions."""

    config = world.config
    item_class = classify(stream.features)
    visible = np.where(stream.revealed, stream.born + config.delay, config.steps)
    n = np.zeros((config.steps, classes), dtype=np.int64)
    k = np.zeros((config.steps, classes), dtype=np.int64)
    for step in range(config.steps):
        seen = visible <= step
        n[step] = np.bincount(item_class[seen], minlength=classes)
        k[step] = np.bincount(item_class[seen & stream.veracity], minlength=classes)
    return n, k


def x0_policy(contract: Mapping[str, Any], world: spread.World, stream: spread.Stream,
              x0: Mapping[str, Any], query_log: list[int]) -> spread.Policy:
    spec = contract["x0"]
    classes = int(spec["class_bins"]) ** int(spec["class_features"])
    classify = spread.item_classes(int(spec["class_bins"]), int(spec["class_features"]))
    archetype_of = spread.archetypes(world, int(spec["threshold_bins"]), cost_bins=world.config.regime == "B")
    n, k = class_aggregates(world, stream, classify, classes)
    if int(n.max()) > int(spec["aggregates"]["n_max"]):
        raise RuntimeError("a class aggregate exceeds the elicited grid")
    state_index = np.asarray([
        [x0["states"][aggregate_state(int(n[t, c]), int(k[t, c]), spec["aggregates"])] for c in range(classes)]
        for t in range(world.config.steps)
    ])
    table = x0["table"]

    def probability(ids: np.ndarray, x: np.ndarray, step: int) -> np.ndarray:
        archetype, item_class = archetype_of[ids], classify(x)
        query_log.append(len(np.unique(archetype * classes + item_class)))
        return core.sigmoid(table[archetype, item_class, state_index[step, item_class]])

    return spread.Policy("X0", probability=probability)


def _learner(d4_contract: Mapping[str, Any], world: spread.World, family: str, geometry_index: int,
             cell: int, replicate: int, device: str, prior: core.LinearPrior) -> spread.Policy:
    """D4 v2's learner with `prior` in place of the hand rule; the same seeds as L0."""

    config = world.config
    components = (int(d4_contract["seeds"]["learner_tag"]), cell, ord(config.regime),
                  d4._family_index(family), geometry_index, replicate)
    observability = d4_contract["world"]["observability"]
    one_row = bool(observability["one_row_per_agent_and_item"])
    if config.regime == "A":
        return spread.pooled_learning_policy(
            family, prior, seed_components=components, label_mode=observability["label_mode"],
            one_row_per_exposure=one_row, device=device,
        )
    return spread.partial_pooling_policy(
        family.removeprefix("partial_"), prior, config.agents, seed_components=components,
        label_mode=observability["label_mode"], device=device, agent_residual="exact",
        one_row_per_exposure=one_row,
    )


def build_world(d4_contract: Mapping[str, Any], profile: str, regime: str, geometry: str, cell: int) -> spread.World:
    """The D4 v2 world of a cell."""

    return d4.build_world(d4_contract, profile, regime, geometry, cell)


def _measured(world: spread.World, stream: spread.Stream, policy: spread.Policy, replicate: int) -> dict[str, Any]:
    return d4.metrics(measure.measured_simulate(world, stream, policy, replicate)[1])


def run_fixed_cell(contract: Mapping[str, Any], d4_contract: Mapping[str, Any], arms: Mapping[str, Any],
                   profile: str, regime: str, geometry: str, cell: int) -> dict[str, Any]:
    """P0 and X0 in one cell; neither depends on the learner family."""

    world = build_world(d4_contract, profile, regime, geometry, cell)
    rows = []
    for replicate in range(int(d4_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        queries: list[int] = []
        policies = {
            "P0": spread.prior_policy("P0", arms["P0"][regime]),
            "X0": x0_policy(contract, world, stream, arms["X0"][regime], queries),
        }
        metrics = {name: _measured(world, stream, policy, replicate) for name, policy in policies.items()}
        metrics["X0"]["queries_without_cache"] = int(sum(queries))
        rows.append({"replicate": replicate, "arms": metrics})
    return {"kind": "fixed", "regime": regime, "geometry": geometry, "cell": cell, "rows": rows}


def run_learner_cell(contract: Mapping[str, Any], d4_contract: Mapping[str, Any], arms: Mapping[str, Any],
                     profile: str, regime: str, family: str, geometry: str, cell: int, device: str) -> dict[str, Any]:
    """L1 in one cell, and L0 again at replicate 0 as a parity check against D4."""

    world = build_world(d4_contract, profile, regime, geometry, cell)
    geometry_index = list(d4_contract["design"]["geometries"]).index(geometry)
    rows = []
    for replicate in range(int(d4_contract["profiles"][profile]["replicates"])):
        stream = spread.build_stream(world, replicate)
        policy = _learner(d4_contract, world, family, geometry_index, cell, replicate, device, arms["P0"][regime])
        rows.append({"replicate": replicate, "arms": {"L1": _measured(world, stream, policy, replicate)}})
    stream = spread.build_stream(world, 0)
    hand = _learner(d4_contract, world, family, geometry_index, cell, 0, device, spread.hand_prior(world.config))
    parity = _measured(world, stream, hand, 0)
    return {"kind": "learner", "regime": regime, "family": family, "geometry": geometry, "cell": cell,
            "rows": rows, "l0_replicate_0": parity}


# -- D4 inputs ----------------------------------------------------------------------


def load_d4_cells(contract: Mapping[str, Any], path: Path, record_path: Path,
                  precommit_sha: str | None) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """D4 v2's development cells, bound by its record.

    The record must come from this precommit (for a registered run), from
    the bound study contract, and from a byte-identical reproduction.
    """

    record = json.loads(record_path.read_text(encoding="utf-8"))
    if precommit_sha is not None and record["precommit_commit"] != precommit_sha:
        raise RuntimeError("the D4 v2 record comes from another precommit")
    if record["contract_sha256"] != _sha256(ROOT / contract["d4"]["contract"]):
        raise RuntimeError("the D4 v2 record comes from another study contract")
    if not all(record["reproduction"]["byte_identical"].values()):
        raise RuntimeError("D4 v2 did not reproduce byte for byte")
    if _sha256(path) != record["reproduction"]["outputs"]["development"]["cells.jsonl"]:
        raise RuntimeError("the D4 v2 cells differ from its record")
    cells = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in cells}


# -- the registered rule ---------------------------------------------------------------


def _label(rule: Mapping[str, Any]) -> str:
    if not rule["material"]:
        return "INDISTINGUISHABLE"
    return "POSITIVE" if rule["mean_cell_delta"] > 0 else "NEGATIVE"


def joined_rows(d4_cell: Mapping[str, Any], fixed: Mapping[str, Any], learner: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for a, b, c in zip(d4_cell["rows"], fixed["rows"], learner["rows"], strict=True):
        if not a["replicate"] == b["replicate"] == c["replicate"]:
            raise RuntimeError("replicates are not aligned")
        rows.append({"replicate": a["replicate"], "arms": {**a["arms"], **b["arms"], **c["arms"]}})
    return rows


def summarize(contract: Mapping[str, Any], d4_contract: Mapping[str, Any],
              d4_cells: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
              cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixed = {(c["regime"], c["geometry"], c["cell"]): c for c in cells if c["kind"] == "fixed"}
    learners = {(c["regime"], c["family"], c["geometry"], c["cell"]): c for c in cells if c["kind"] == "learner"}
    parity_failures = [
        f"{r}/{f}/{g}/{cell}" for (r, f, g, cell), c in sorted(learners.items())
        if c["l0_replicate_0"] != d4_cells[(r, f, g, cell)]["rows"][0]["arms"]["L0"]
    ]
    groups: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = {}
    for (regime, family, geometry, cell), learner in sorted(learners.items()):
        rows = joined_rows(d4_cells[(regime, family, geometry, cell)], fixed[(regime, geometry, cell)], learner)
        groups.setdefault((regime, family, geometry), []).append(rows)
    tag = int(contract["seeds"]["rule_tag"])
    table: dict[str, Any] = {}
    every = {**COMPARISONS, **DESCRIPTIVE}
    arms = (*d4.ARMS, *ARMS)
    for g_index, ((regime, family, geometry), members) in enumerate(sorted(groups.items())):
        entry: dict[str, Any] = {"cells": len(members)}
        for m_index, measure_name in enumerate(d4.MEASURES):
            rules = {}
            for c_index, (name, (left, right)) in enumerate(every.items()):
                means = [float(np.mean([r["arms"][left][measure_name] - r["arms"][right][measure_name]
                                        for r in rows])) for rows in members]
                if not np.all(np.isfinite(means)):
                    raise RuntimeError(f"a {measure_name} is undefined in {regime}/{family}/{geometry}")
                rules[name] = d4._group_rule(contract, means, (tag, g_index, m_index, c_index))
            entry[measure_name] = rules
            entry[f"mean_{measure_name}"] = {
                arm: round(float(np.mean([r["arms"][arm][measure_name] for rows in members for r in rows])), 4)
                for arm in arms
            }
        entry["mean_false_reach"] = {
            arm: round(float(np.mean([r["arms"][arm]["false_reach"] for rows in members for r in rows])), 2)
            for arm in arms
        }
        entry["x0_queries_without_cache_per_run"] = round(float(np.mean(
            [r["arms"]["X0"]["queries_without_cache"] for rows in members for r in rows])), 1)
        table[f"{regime}/{family}/{geometry}"] = entry
    primary = {key: entry[d4.PRIMARY] for key, entry in table.items()}
    return {"table": table, "parity_failures": parity_failures,
            "dispositions": dispositions(contract, d4_contract, primary)}


def dispositions(contract: Mapping[str, Any], d4_contract: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, Any]:
    design = d4_contract["design"]
    thresholds = contract["dispositions"]
    study = {k: v for k, v in table.items() if k.split("/")[2] in design["study_geometries"]}
    labels = {name: {k: _label(v[name]) for k, v in study.items()} for name in COMPARISONS}
    counts = {
        name: {regime: {label: sum(1 for k, v in by_group.items() if k.startswith(f"{regime}/") and v == label)
                        for label in LABELS} for regime in design["regimes"]}
        for name, by_group in labels.items()
    }
    at_least = int(thresholds["mostly_at_least"])
    summary = {
        name: {regime: ("MOSTLY_POSITIVE" if c["POSITIVE"] >= at_least
                        else "MOSTLY_NEGATIVE" if c["NEGATIVE"] >= at_least else "MIXED_OR_NONE")
               for regime, c in by_regime.items()}
        for name, by_regime in counts.items()
    }
    null_violations = [
        k for k, v in table.items()
        if k.split("/")[2] == design["null_geometry"]
        and v["closed_loop_on_llm_prior"]["material"] and v["closed_loop_on_llm_prior"]["mean_cell_delta"] > 0
    ]
    return {
        "design_check": "DESIGN_DEFECT_IN_NULL_WORLD" if null_violations else "NULL_WORLD_PASSES",
        "null_violations": null_violations,
        "by_comparison": summary, "counts": counts, "labels_by_group": labels,
        "study_groups_per_regime": len(study) // len(design["regimes"]),
    }


# -- main ------------------------------------------------------------------------------


def _fixed_task(args: tuple[str, str, Path, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, profile, elicitation_dir, regime, geometry, cell = args
    contract = load_contract(Path(contract_path))
    d4_contract = load_d4_contract(contract)
    arms = build_llm_arms(contract, load_elicitation(elicitation_dir, contract))
    started = time.perf_counter()
    result = run_fixed_cell(contract, d4_contract, arms, profile, regime, geometry, cell)
    return f"fixed/{regime}/{geometry}/{cell}", result, {"device": "cpu", "task_seconds": round(time.perf_counter() - started, 1)}


def _learner_task(args: tuple[str, str, Path, str, str, str, int]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    contract_path, profile, elicitation_dir, regime, family, geometry, cell = args
    contract = load_contract(Path(contract_path))
    d4_contract = load_d4_contract(contract)
    d4.apply_learner_settings(d4_contract)
    arms = build_llm_arms(contract, load_elicitation(elicitation_dir, contract))
    device = d4.device_for(family)
    started = time.perf_counter()
    result = run_learner_cell(contract, d4_contract, arms, profile, regime, family, geometry, cell, device)
    return (f"learner/{regime}/{family}/{geometry}/{cell}", result,
            {"device": device, "task_seconds": round(time.perf_counter() - started, 1)})


def _host() -> dict[str, Any]:
    return d4._host()


def run(args: argparse.Namespace, contract: Mapping[str, Any]) -> None:
    d4_contract = load_d4_contract(contract)
    is_registered_run = args.profile == "development" and args.pilot_cells is None
    precommit = verify_precommit(args.precommit_sha, contract) if is_registered_run else None
    design = d4_contract["design"]
    cells_per_group = args.pilot_cells or int(d4_contract["profiles"][args.profile]["cells"])
    learner_specs = [
        (r, f, g, cell) for r in design["regimes"] for f in design["families"][r] for g in design["geometries"]
        for cell in range(cells_per_group)
    ]
    learner_specs.sort(key=lambda s: "mlp" not in s[1])  # heaviest first, for load balance
    fixed_specs = [(r, g, cell) for r in design["regimes"] for g in design["geometries"] for cell in range(cells_per_group)]
    base = (str(args.contract), args.profile, args.elicitation_dir)
    jobs: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [(_learner_task, (*base, *s)) for s in learner_specs]
    jobs += [(_fixed_task, (*base, *s)) for s in fixed_specs]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.jsonl"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(function, job) for function, job in jobs]
        for future in as_completed(futures):
            name, cell, timing = future.result()
            results.append(cell)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"done": len(results), "of": len(jobs), "task": name, **timing,
                                         "elapsed_seconds": round(time.perf_counter() - started, 1)}) + "\n")
    ordered = sorted(results, key=lambda c: (c["kind"], c["regime"], c.get("family", ""), c["geometry"], c["cell"]))
    cells_path = args.output_dir / "cells.jsonl"
    cells_path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in ordered), encoding="utf-8")
    summary: dict[str, Any] = {
        "label": contract["label"],
        "status": "STUDY_RESULT" if is_registered_run else "NOT_FOR_RESULTS_SMOKE_OR_PILOT",
        "profile": args.profile, "cells_per_group": cells_per_group,
        "contract_sha256": _sha256(args.contract), "precommit_commit": precommit["commit_sha"] if precommit else None,
        "elicitation_sha256": _sha256(args.elicitation_dir / "elicitation.jsonl"),
        "llm_arms": build_llm_arms(contract, load_elicitation(args.elicitation_dir, contract))["report"],
    }
    if args.d4_cells is not None:
        if args.d4_record is None:
            raise RuntimeError("--d4-cells needs --d4-record")
        d4_cells = load_d4_cells(contract, args.d4_cells, args.d4_record, precommit["commit_sha"] if precommit else None)
        summary.update(summarize(contract, d4_contract, d4_cells, ordered))
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "precommit": precommit, "host": _host(), "workers": args.workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"cells.jsonl": _sha256(cells_path), "summary.json": _sha256(summary_path)},
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"dispositions": summary.get("dispositions", {}).get("by_comparison"),
                      "parity_failures": summary.get("parity_failures"), **receipt["outputs"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("elicit", "run"))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-prompts", type=int, default=None,
                        help="elicit: only this many prompts per set; never read for results")
    parser.add_argument("--elicitation-dir", type=Path)
    parser.add_argument("--d4-cells", type=Path, help="run: D4 v2's development cells.jsonl")
    parser.add_argument("--d4-record", type=Path, help="run: D4 v2's record JSON")
    parser.add_argument("--profile", choices=("development", "smoke"), default="development")
    parser.add_argument("--pilot-cells", type=int, default=None,
                        help="run: a full-load pilot with this many cells per group; never read for results")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    if args.command == "elicit":
        registered = args.smoke_prompts is None
        precommit = verify_precommit(args.precommit_sha, contract) if registered else None
        result = elicit(contract, limit=args.smoke_prompts)
        receipt = write_elicitation(args.output_dir, result, contract, args.contract, precommit)
        print(json.dumps({"validity": receipt["validity"], "determinism": receipt["determinism"],
                          "timing_seconds": receipt["timing_seconds"], **receipt["outputs"]}))
        return
    if args.elicitation_dir is None:
        parser.error("run needs --elicitation-dir")
    run(args, contract)


if __name__ == "__main__":
    main()
