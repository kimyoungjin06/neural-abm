#!/usr/bin/env python3
"""N11a v2: can the empirical model ladder separate learning from anticipation?

v1 passed identification, but entrants drawn from the start toward the
episode's eventual success relation, without learning, also won. Learning can
act only after evidence is visible; anticipation acts from the first quarter.
v2 adds an anticipatory competitor, B7, that applies the episode's eventual
success relation from the first quarter, and requires NABM to beat it. The v1
simulator, ladder rows, learners, and bootstrap are reused unchanged, verified
by hash. No empirical row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_ladder_recovery_v2.yaml"
V1_SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_ladder_recovery_v1.py"
V1_SCRIPT_SHA256 = "3ac85cf8a55c8b8ec695c8543c0cf1a6ad114b0f5b42607dcde73decaa415102"
ANTICIPATORY = "B7_anticipatory"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v1() -> ModuleType:
    if _sha256(V1_SCRIPT) != V1_SCRIPT_SHA256:
        raise RuntimeError("the v1 N11a script drifted from its bound hash")
    spec = importlib.util.spec_from_file_location("n11a_v1", V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_v1()
MODELS = (*v1.MODELS, ANTICIPATORY)
NABM = v1.NABM
NON_LEARNING = (*v1.NON_LEARNING, ANTICIPATORY)


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("the N11a v2 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_empirical_ladder_recovery.v2":
        raise ValueError("unexpected N11a v2 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("the N11a v2 contract is not a precommit design")
    amends = contract["amends"]
    for key in ("contract_v1", "script_v1", "result_v1"):
        if _sha256(ROOT / amends[key]["path"]) != amends[key]["sha256"]:
            raise ValueError(f"the amended {key} drifted from its bound hash")
    original = yaml.safe_load((ROOT / amends["contract_v1"]["path"]).read_text(encoding="utf-8"))
    for key in ("episode_structure", "mechanisms", "reported_sign_test", "profiles"):
        if contract[key] != original[key]:
            raise ValueError(f"v2 may not change the v1 {key} block")
    for key in ("learners", "fit", "selection", "rows", "shared_inputs"):
        if contract["ladder"][key] != original["ladder"][key]:
            raise ValueError(f"v2 may not change the v1 ladder {key}")
    if contract["decision"]["bootstrap"] != original["decision"]["bootstrap"]:
        raise ValueError("v2 may not change the bootstrap")
    if tuple(contract["ladder"]["models"]) != MODELS or tuple(contract["ladder"]["non_learning"]) != NON_LEARNING:
        raise ValueError("the v2 ladder must be v1's plus the anticipatory competitor")
    roles = {w["id"]: w["role"] for w in contract["worlds"]}
    if roles.get("W6_static_aligned_drift") != "MUST_NOT_FALSE_RECOVER":
        raise ValueError("the anticipation world must be must-not-falsely-recover in v2")
    for world in contract["worlds"]:
        if world["role"] == "MUST_NOT_FALSE_RECOVER" and world["kappa"] != 0:
            raise ValueError("a must-not-falsely-recover world must not learn")
        if world["role"] == "MUST_RECOVER" and world["kappa"] <= 0:
            raise ValueError("a must-recover world must learn")
    if int(contract["seeds"]["base_seed"]) in set(contract["seeds"]["opened"]):
        raise ValueError("v2 must use fresh seeds")
    v1.validate_contract(original)


def anticipatory_rows(episode: Mapping[str, Any], minimum: int) -> np.ndarray:
    """The episode's eventual success relation, applied to every quarter's risk set from the first quarter."""

    x, entry, observed = episode["x"], episode["entry"], episode["observed"]
    entrants = np.flatnonzero(entry >= 0)
    labels = observed[entrants, : v1.EVIDENCE_QUARTERS].any(axis=1).astype(int)
    terms = []
    for quarter in range(v1.QUARTERS):
        at_risk = np.flatnonzero((entry == -1) | (entry >= quarter))
        term = v1.learned_term("linear", {"C": 1.0}, x[entrants], labels, x[at_risk], 0, minimum)
        terms.append(term if term is not None else np.zeros(at_risk.size))
    return np.concatenate(terms)


def design(model: str, rows: Mapping[str, np.ndarray]) -> np.ndarray:
    if model == ANTICIPATORY:
        return np.hstack([rows["attention"][:, None], rows["x"], rows[ANTICIPATORY][:, None]])
    return v1.design(model, rows)


def fit(model: str, rows: Mapping[str, np.ndarray]) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(design(model, rows), rows["y"])


def episode_loglik(clf: LogisticRegression, model: str, rows: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    p = np.clip(clf.predict_proba(design(model, rows))[:, 1], 1e-12, 1 - 1e-12)
    ll = rows["y"] * np.log(p) + (1 - rows["y"]) * np.log(1 - p)
    return np.bincount(rows["episode"], weights=ll, minlength=count)


def run_task(contract: Mapping[str, Any], profile: str, world_index: int, replicate: int) -> dict[str, Any]:
    params = v1.profile_params(contract, profile)
    world = contract["worlds"][world_index]
    mechanisms = contract["mechanisms"]
    minimum = int(mechanisms["evidence"]["minimum_visible_entrants"])
    count = int(params["episodes_per_replicate"])
    discovery = int(params["discovery_episodes"])
    base = np.random.SeedSequence((int(contract["seeds"]["base_seed"]), world_index, replicate))
    episode_seeds = base.spawn(count + 1)
    with threadpool_limits(limits=1):
        episodes = [v1.simulate_episode(world, params, mechanisms, seq.spawn(2)[0]) for seq in episode_seeds[:count]]
        rows = []
        for index, (episode, seq) in enumerate(zip(episodes, episode_seeds[:count], strict=True)):
            part = v1.ladder_rows(episode, contract["ladder"], minimum, seq.spawn(2)[1])
            part[ANTICIPATORY] = anticipatory_rows(episode, minimum)
            part["episode"] = np.full(part["y"].size, index)
            rows.append(part)
        disc = v1._concat(rows[:discovery])
        conf = v1._concat(rows[discovery:])
        fitted = {model: fit(model, disc) for model in MODELS}
        disc_ll = {model: float(episode_loglik(fitted[model], model, disc, count).sum()) for model in MODELS}
        conf_ll = {model: episode_loglik(fitted[model], model, conf, count)[discovery:] for model in MODELS}
    best_nabm = max(NABM, key=lambda m: (disc_ll[m], m))
    best_other = max(NON_LEARNING, key=lambda m: (disc_ll[m], m))
    decision = contract["decision"]["bootstrap"]
    rng = np.random.default_rng(episode_seeds[count])
    comparisons = {}
    for name, rival in (("best_non_learning", best_other), ("b2", "B2"), ("anticipatory", ANTICIPATORY)):
        difference = conf_ll[best_nabm] - conf_ll[rival]
        lower = v1._bootstrap_lower(difference, int(decision["resamples"]), float(decision["lower_quantile"]), rng)
        comparisons[name] = {"rival": rival, "mean": float(difference.mean()), "lower_bound": lower,
                             "won": bool(difference.mean() > 0 and lower > 0)}
    sign = v1.sign_test(episodes[discovery:], float(contract["reported_sign_test"]["threshold"]), int(decision["resamples"]), rng)
    return {
        "world": world["id"],
        "role": world["role"],
        "replicate": replicate,
        "best_nabm": best_nabm,
        "best_non_learning": best_other,
        "comparisons": comparisons,
        "nabm_win": all(c["won"] for c in comparisons.values()),
        "learned_coefficient": float(fitted[best_nabm].coef_[0][-1]),
        "anticipatory_coefficient": float(fitted[ANTICIPATORY].coef_[0][-1]),
        "mean_episode_ll_difference": comparisons["best_non_learning"]["mean"],
        "discovery_ll": disc_ll,
        "confirmatory_ll": {model: float(values.sum()) for model, values in conf_ll.items()},
        "sign_test": sign,
        "entrants_per_episode_median": float(np.median([int((e["entry"] >= 0).sum()) for e in episodes])),
    }


def _task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_task(*args)


def run_study(contract: Mapping[str, Any], profile: str, workers: int) -> list[dict[str, Any]]:
    replicates = int(contract["profiles"][profile]["replicates"])
    tasks = [(contract, profile, w, r) for w in range(len(contract["worlds"])) for r in range(replicates)]
    if workers <= 1:
        return [_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_task, tasks))


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    result = v1.decide(rows, contract)
    for world_id, world in result["worlds"].items():
        selected = [r for r in rows if r["world"] == world_id]
        world["beats_anticipatory"] = sum(r["comparisons"]["anticipatory"]["won"] for r in selected)
        world["median_learned_coefficient"] = float(np.median([r["learned_coefficient"] for r in selected])) if selected else None
        world["median_anticipatory_coefficient"] = float(np.median([r["anticipatory_coefficient"] for r in selected])) if selected else None
    return result


def write_outputs(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], precommit: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    markdown = v1._markdown(summary).replace("ladder recovery v1 (N11a)", "ladder recovery v2 (N11a v2)")
    files = {
        "results.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": markdown,
    }
    for name, content in files.items():
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    receipt = {
        "schema_version": "pivot_empirical_ladder_recovery_receipt.v2",
        "status": "COMPLETE",
        "disposition": summary["decision"]["disposition"],
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {name: _sha256(output_dir / name) for name in files},
        "empirical_rows_read": 0,
    }
    with (output_dir / "receipt.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps({"status": "PRECOMMIT_PLAN_VALID", "contract_sha256": _sha256(config_path),
                          "worlds": [w["id"] for w in contract["worlds"]], "source_bindings": v1.verify_source_bindings(contract)},
                         indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or args.output_dir is None:
        parser.error("run requires --precommit-sha and --output-dir")
    precommit = v1.verify_precommit(args.precommit_sha, contract)
    rows = run_study(contract, args.profile, args.workers)
    summary = {"profile": args.profile, "precommit_sha": args.precommit_sha, "decision": decide(rows, contract),
               "boundaries": contract["boundaries"]}
    receipt = write_outputs(args.output_dir.resolve(), rows, summary, precommit, config_path)
    print(json.dumps({"status": receipt["status"], "disposition": receipt["disposition"]}))


if __name__ == "__main__":
    main()
