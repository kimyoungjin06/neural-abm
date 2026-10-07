#!/usr/bin/env python3
"""N11a: does the empirical model ladder recover learned entry selection?

Synthetic worlds mirror the empirical episodes: an at-risk population, twelve
response quarters, an attention surge, delayed and partly observed outcomes.
In learning worlds, entry selectivity follows a success predictor learned from
the outcomes that have become visible. In no-learning worlds, entry is static,
and the confounds that fool a sign test are added: topic growth, linkage that
improves over time, and outcome-independent drift in selectivity. The ladder
is fitted on discovery episodes and scored on confirmatory ones. No empirical
row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import re
import subprocess
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_ladder_recovery_v1.yaml"
N10_V2 = ROOT / "experiments" / "configs" / "pivot_empirical_outcome_selection_trend_v2.yaml"
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
MODELS = ("B1", "B2", "B2t", "B3", "B4_linear", "B5_tree", "B6_mlp")
NABM = ("B4_linear", "B5_tree", "B6_mlp")
NON_LEARNING = ("B1", "B2", "B2t", "B3")
LEARNERS = {"B4_linear": "linear", "B5_tree": "tree", "B6_mlp": "mlp"}
FEATURES = 4
QUARTERS = 12
RESPONSE_YEARS = 3
EVIDENCE_QUARTERS = 4
PERSISTENCE_QUARTERS = 12


# -- contract ----------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("the N11a contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_empirical_ladder_recovery.v1":
        raise ValueError("unexpected N11a schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("the N11a contract is not a precommit design")
    ladder = contract["ladder"]
    if tuple(ladder["models"]) != MODELS or tuple(ladder["nabm"]) != NABM or tuple(ladder["non_learning"]) != NON_LEARNING:
        raise ValueError("the ladder changed")
    roles = {world["id"]: world["role"] for world in contract["worlds"]}
    if sorted(r for r in roles.values() if r == "MUST_RECOVER") != ["MUST_RECOVER"] * 2:
        raise ValueError("two worlds must be registered as must-recover")
    if sum(r == "MUST_NOT_FALSE_RECOVER" for r in roles.values()) != 3:
        raise ValueError("three worlds must be registered as must-not-false-recover")
    for world in contract["worlds"]:
        if world["role"] == "MUST_RECOVER" and world["kappa"] <= 0:
            raise ValueError("a must-recover world must learn")
        if world["role"] != "MUST_RECOVER" and world["role"] != "REPORTED_POWER" and world["kappa"] != 0:
            raise ValueError("a no-learning world must not learn")
    n10 = yaml.safe_load(N10_V2.read_text(encoding="utf-8"))
    if float(contract["reported_sign_test"]["threshold"]) != float(n10["threshold_transfer"]["fine_clock_threshold"]):
        raise ValueError("the reported sign test must use the N10 v2 data threshold")
    if n10["threshold_transfer"]["empirical_entry_levels"] != QUARTERS:
        raise ValueError("the quarter clock differs from N10 v2")
    structure = contract["episode_structure"]
    if structure["quarters"] != QUARTERS or len(structure["features"]) != FEATURES:
        raise ValueError("the episode structure changed")


def profile_params(contract: Mapping[str, Any], profile: str) -> dict[str, Any]:
    return {**contract["episode_structure"], **contract["profiles"][profile]}


def verify_source_bindings(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    verified = []
    for binding in contract["source_bindings"]:
        actual = _sha256(ROOT / binding["path"])
        if actual != binding["sha256"]:
            raise RuntimeError(f"source binding mismatch for {binding['path']}")
        verified.append({"path": binding["path"], "sha256": actual, "role": binding["role"]})
    return verified


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    if FULL_SHA_PATTERN.fullmatch(commit_sha) is None:
        raise RuntimeError("run requires a full 40-character lowercase Git SHA")
    if _git("rev-parse", "HEAD") != commit_sha:
        raise RuntimeError("precommit SHA is not HEAD")
    paths = [str(path) for path in contract["precommit"]["precommit_paths"]]
    if _git("status", "--porcelain", "--untracked-files=no", "--", *paths):
        raise RuntimeError("precommit paths are dirty")
    tracked = set(_git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    if sorted(set(paths) - tracked):
        raise RuntimeError("precommit paths missing from commit")
    return {"commit_sha": commit_sha, "precommit_paths": paths, "source_bindings": verify_source_bindings(contract)}


# -- the synthetic episodes --------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def attention_curve() -> np.ndarray:
    quarters = np.arange(QUARTERS, dtype=float)
    return np.exp(-((quarters - 3.0) ** 2) / (2 * 3.0**2))


def draw_features(rng: np.random.Generator, n: int, correlation: float) -> np.ndarray:
    covariance = (1 - correlation) * np.eye(FEATURES) + correlation * np.ones((FEATURES, FEATURES))
    return rng.multivariate_normal(np.zeros(FEATURES), covariance, size=n)


def linkage(quarter: np.ndarray, confounded: bool) -> np.ndarray:
    if not confounded:
        return np.ones_like(quarter, dtype=float)
    return 0.70 + 0.25 * np.minimum(quarter, 23) / 23


def centred_logit(probability: np.ndarray) -> np.ndarray:
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    return logit - logit.mean()


def learned_term(family: str, settings: Mapping[str, Any], x_visible: np.ndarray, y_visible: np.ndarray,
                 x_eval: np.ndarray, seed: int, minimum: int) -> np.ndarray | None:
    """The centred logit of a success predictor learned from visible evidence, or None without enough evidence."""

    if y_visible.size < minimum or np.unique(y_visible).size < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if family == "linear":
            model = LogisticRegression(C=float(settings["C"]), max_iter=1000)
        elif family == "tree":
            model = GradientBoostingClassifier(n_estimators=int(settings["n_estimators"]), max_depth=int(settings["max_depth"]),
                                               learning_rate=float(settings["learning_rate"]), random_state=seed)
        elif family == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(int(settings["hidden"]),), alpha=float(settings["alpha"]),
                                  max_iter=int(settings["max_iter"]), random_state=seed)
        else:
            raise ValueError(f"unknown learner {family}")
        model.fit(x_visible, y_visible)
    return centred_logit(model.predict_proba(x_eval)[:, 1])


def _drift(world: Mapping[str, Any], quarter: int, weights: np.ndarray, mean: np.ndarray) -> np.ndarray:
    drift = np.zeros(FEATURES)
    scale = 0.8 * quarter / (QUARTERS - 1)
    if "independent_drift" in world["confounds"]:
        direction = np.array([0.5, 0.5, -0.5, 0.5])
        drift += scale * direction / np.linalg.norm(direction)
    if "aligned_drift" in world["confounds"]:
        direction = weights - mean
        drift += scale * direction / max(np.linalg.norm(direction), 1e-9)
    return drift


def simulate_episode(world: Mapping[str, Any], params: Mapping[str, Any], mechanisms: Mapping[str, Any], seed_seq: np.random.SeedSequence) -> dict[str, Any]:
    entry_rng, outcome_rng, learner_seeds = (np.random.default_rng(s) for s in seed_seq.spawn(3))
    n = int(params["at_risk_authors"])
    x = draw_features(entry_rng, n, float(params["feature_correlation"]))
    success = mechanisms["success"]
    weight_mean = np.asarray(success["episode_weight_mean"], dtype=float)
    weights = entry_rng.normal(weight_mean, float(success["episode_weight_sd"]))
    static = mechanisms["static_entry"]
    beta = np.asarray(static["beta"], dtype=float)
    attention = attention_curve()
    growth = "growth" in world["confounds"]
    linked = "linkage" in world["confounds"]
    follow = mechanisms["follow_up"]
    evidence = mechanisms["evidence"]
    kappa = float(world["kappa"])
    entry = np.full(n, -1, dtype=int)
    observed = np.zeros((n, PERSISTENCE_QUARTERS), dtype=bool)
    for quarter in range(QUARTERS):
        at_risk = np.flatnonzero(entry == -1)
        logit = float(static["a0"]) + float(static["a1"]) * attention[quarter] + x[at_risk] @ (beta + _drift(world, quarter, weights, weight_mean))
        if kappa > 0:
            visible = np.flatnonzero((entry >= 0) & (entry <= quarter - EVIDENCE_QUARTERS - 1))
            labels = observed[visible, :EVIDENCE_QUARTERS].any(axis=1).astype(int)
            term = learned_term("linear", {"C": 1.0}, x[visible], labels, x[at_risk], int(learner_seeds.integers(2**31)),
                                int(evidence["minimum_visible_entrants"]))
            if term is not None:
                logit = logit + kappa * term
        entered = at_risk[entry_rng.random(at_risk.size) < _sigmoid(logit)]
        entry[entered] = quarter
        success_logit = float(success["c0"]) + x[entered] @ weights + (1.0 * quarter / (QUARTERS - 1) if growth else 0.0)
        succeeded = outcome_rng.random(entered.size) < _sigmoid(success_logit)
        rate = np.where(succeeded, float(follow["success_rate_per_quarter"]), float(follow["failure_rate_per_quarter"]))
        true_follow = outcome_rng.random((entered.size, PERSISTENCE_QUARTERS)) < rate[:, None]
        calendar = quarter + np.arange(1, PERSISTENCE_QUARTERS + 1)
        seen = outcome_rng.random((entered.size, PERSISTENCE_QUARTERS)) < linkage(calendar, linked)[None, :]
        observed[entered] = true_follow & seen
    return {"x": x, "entry": entry, "observed": observed, "attention": attention}


# -- ladder rows -------------------------------------------------------------------------


def ladder_rows(episode: Mapping[str, Any], ladder: Mapping[str, Any], minimum: int, seed_seq: np.random.SeedSequence) -> dict[str, np.ndarray]:
    """One Bernoulli row per at-risk author and quarter, with every model's inputs."""

    x, entry, observed, attention = episode["x"], episode["entry"], episode["observed"], episode["attention"]
    early = observed[:, :EVIDENCE_QUARTERS].any(axis=1).astype(int)
    seeds = {name: np.random.default_rng(s) for name, s in zip(LEARNERS, seed_seq.spawn(len(LEARNERS)), strict=True)}
    columns: dict[str, list[np.ndarray]] = {key: [] for key in ("y", "attention", "x", "year", "has_ev", "rate", *LEARNERS)}
    for quarter in range(QUARTERS):
        at_risk = np.flatnonzero((entry == -1) | (entry >= quarter))
        visible = np.flatnonzero((entry >= 0) & (entry <= quarter - EVIDENCE_QUARTERS - 1))
        has_evidence = visible.size >= minimum
        columns["y"].append((entry[at_risk] == quarter).astype(float))
        columns["attention"].append(np.full(at_risk.size, attention[quarter]))
        columns["x"].append(x[at_risk])
        columns["year"].append(np.full(at_risk.size, quarter // 4))
        columns["has_ev"].append(np.full(at_risk.size, float(has_evidence)))
        columns["rate"].append(np.full(at_risk.size, float(early[visible].mean()) if has_evidence else 0.0))
        for model, family in LEARNERS.items():
            term = learned_term(family, ladder["learners"][family], x[visible], early[visible], x[at_risk],
                                int(seeds[model].integers(2**31)), minimum)
            columns[model].append(term if term is not None else np.zeros(at_risk.size))
    return {key: np.concatenate(values) for key, values in columns.items()}


def design(model: str, rows: Mapping[str, np.ndarray]) -> np.ndarray:
    attention = rows["attention"][:, None]
    x = rows["x"]
    if model == "B1":
        return attention
    if model == "B2":
        return np.hstack([attention, x])
    if model == "B2t":
        return np.hstack([attention] + [x * (rows["year"] == year)[:, None] for year in range(RESPONSE_YEARS)])
    if model == "B3":
        return np.hstack([attention, x, rows["has_ev"][:, None], (rows["has_ev"] * rows["rate"])[:, None]])
    if model in LEARNERS:
        return np.hstack([attention, x, rows[model][:, None]])
    raise ValueError(f"unknown model {model}")


def fit(model: str, rows: Mapping[str, np.ndarray]) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(design(model, rows), rows["y"])


def episode_loglik(clf: LogisticRegression, model: str, rows: Mapping[str, np.ndarray], episode: np.ndarray, count: int) -> np.ndarray:
    p = np.clip(clf.predict_proba(design(model, rows))[:, 1], 1e-12, 1 - 1e-12)
    ll = rows["y"] * np.log(p) + (1 - rows["y"]) * np.log(1 - p)
    return np.bincount(episode, weights=ll, minlength=count)


def _concat(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _bootstrap_lower(values: np.ndarray, resamples: int, quantile: float, rng: np.random.Generator) -> float:
    index = rng.integers(0, values.size, size=(resamples, values.size))
    return float(np.quantile(values[index].mean(axis=1), quantile))


def sign_test(episodes: Sequence[Mapping[str, Any]], threshold: float, resamples: int, rng: np.random.Generator) -> dict[str, Any]:
    """The reported N10 v2 statistic: per-episode Spearman of entry quarter with twelve-quarter persistence."""

    values = []
    for episode in episodes:
        entrants = np.flatnonzero(episode["entry"] >= 0)
        offsets = episode["entry"][entrants]
        persisted = episode["observed"][entrants].any(axis=1).astype(float)
        if entrants.size >= 3 and np.unique(offsets).size >= 2 and np.unique(persisted).size >= 2:
            values.append(float(spearmanr(offsets, persisted).statistic))
    array = np.asarray(values)
    if array.size == 0:
        return {"mean": None, "lower_bound": None, "positive": False, "episodes": 0}
    mean = float(array.mean())
    lower = _bootstrap_lower(array, resamples, 0.05, rng)
    return {"mean": mean, "lower_bound": lower, "positive": bool(mean > threshold and lower > 0), "episodes": int(array.size)}


def run_task(contract: Mapping[str, Any], profile: str, world_index: int, replicate: int) -> dict[str, Any]:
    params = profile_params(contract, profile)
    world = contract["worlds"][world_index]
    mechanisms = contract["mechanisms"]
    minimum = int(mechanisms["evidence"]["minimum_visible_entrants"])
    count = int(params["episodes_per_replicate"])
    discovery = int(params["discovery_episodes"])
    base = np.random.SeedSequence((int(contract["seeds"]["base_seed"]), world_index, replicate))
    episode_seeds = base.spawn(count + 1)
    with threadpool_limits(limits=1):
        episodes = [simulate_episode(world, params, mechanisms, seq.spawn(2)[0]) for seq in episode_seeds[:count]]
        rows = []
        for index, (episode, seq) in enumerate(zip(episodes, episode_seeds[:count], strict=True)):
            part = ladder_rows(episode, contract["ladder"], minimum, seq.spawn(2)[1])
            part["episode"] = np.full(part["y"].size, index)
            rows.append(part)
        disc = _concat(rows[:discovery])
        conf = _concat(rows[discovery:])
        fitted = {model: fit(model, disc) for model in MODELS}
        disc_ll = {model: float(episode_loglik(fitted[model], model, disc, disc["episode"], count).sum()) for model in MODELS}
        conf_ll = {model: episode_loglik(fitted[model], model, conf, conf["episode"], count)[discovery:] for model in MODELS}
    best_nabm = max(NABM, key=lambda m: (disc_ll[m], m))
    best_other = max(NON_LEARNING, key=lambda m: (disc_ll[m], m))
    difference = conf_ll[best_nabm] - conf_ll[best_other]
    nested = conf_ll[best_nabm] - conf_ll["B2"]
    decision = contract["decision"]["bootstrap"]
    rng = np.random.default_rng(episode_seeds[count])
    lower = _bootstrap_lower(difference, int(decision["resamples"]), float(decision["lower_quantile"]), rng)
    nested_lower = _bootstrap_lower(nested, int(decision["resamples"]), float(decision["lower_quantile"]), rng)
    mean = float(difference.mean())
    sign = sign_test(episodes[discovery:], float(contract["reported_sign_test"]["threshold"]), int(decision["resamples"]), rng)
    entrants = [int((e["entry"] >= 0).sum()) for e in episodes]
    return {
        "world": world["id"],
        "role": world["role"],
        "replicate": replicate,
        "best_nabm": best_nabm,
        "best_non_learning": best_other,
        "mean_episode_ll_difference": mean,
        "lower_bound": lower,
        "mean_episode_ll_difference_vs_b2": float(nested.mean()),
        "lower_bound_vs_b2": nested_lower,
        "nabm_win": bool(mean > 0 and lower > 0 and nested.mean() > 0 and nested_lower > 0),
        "learned_coefficient": float(fitted[best_nabm].coef_[0][-1]),
        "discovery_ll": disc_ll,
        "confirmatory_ll": {model: float(values.sum()) for model, values in conf_ll.items()},
        "sign_test": sign,
        "entrants_per_episode_median": float(np.median(entrants)),
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


# -- decision ----------------------------------------------------------------------------


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    rule = contract["decision"]
    worlds = {}
    for world in contract["worlds"]:
        selected = [r for r in rows if r["world"] == world["id"]]
        wins = sum(r["nabm_win"] for r in selected)
        if world["role"] == "MUST_RECOVER":
            ok = wins >= int(rule["must_recover_at_least"])
        elif world["role"] == "MUST_NOT_FALSE_RECOVER":
            ok = wins <= int(rule["must_not_false_recover_at_most"])
        else:
            ok = None
        worlds[world["id"]] = {
            "role": world["role"],
            "nabm_wins": wins,
            "replicates": len(selected),
            "meets_role": ok,
            "sign_test_positive": sum(r["sign_test"]["positive"] for r in selected),
            "best_nabm": sorted({r["best_nabm"] for r in selected}),
            "best_non_learning": sorted({r["best_non_learning"] for r in selected}),
            "median_ll_difference": float(np.median([r["mean_episode_ll_difference"] for r in selected])) if selected else None,
        }
    registered = [w for w in worlds.values() if w["meets_role"] is not None]
    passed = all(w["meets_role"] for w in registered)
    return {"disposition": "IDENTIFICATION_PASSES" if passed else "IDENTIFICATION_FAILS", "worlds": worlds}


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot empirical ladder recovery v1 (N11a)",
        "",
        f"Profile: `{summary['profile']}`. Disposition: `{summary['decision']['disposition']}`",
        "",
        "| World | Role | NABM wins | Sign test positive | Best NABM | Best non-learning |",
        "|---|---|---:|---:|---|---|",
    ]
    for world_id, w in summary["decision"]["worlds"].items():
        lines.append(f"| {world_id} | {w['role']} | {w['nabm_wins']}/{w['replicates']} | {w['sign_test_positive']}/{w['replicates']} | "
                     f"{', '.join(w['best_nabm'])} | {', '.join(w['best_non_learning'])} |")
    lines += ["", "Synthetic worlds only. No empirical row was read.", ""]
    return "\n".join(lines)


def write_outputs(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], precommit: Mapping[str, Any], contract_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "results.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in files.items():
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    receipt = {
        "schema_version": "pivot_empirical_ladder_recovery_receipt.v1",
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
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps({"status": "PRECOMMIT_PLAN_VALID", "contract_sha256": _sha256(config_path),
                          "worlds": [w["id"] for w in contract["worlds"]], "source_bindings": verify_source_bindings(contract)},
                         indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or args.output_dir is None:
        parser.error("run requires --precommit-sha and --output-dir")
    precommit = verify_precommit(args.precommit_sha, contract)
    rows = run_study(contract, args.profile, args.workers)
    summary = {"profile": args.profile, "precommit_sha": args.precommit_sha, "decision": decide(rows, contract),
               "boundaries": contract["boundaries"]}
    receipt = write_outputs(args.output_dir.resolve(), rows, summary, precommit, config_path)
    print(json.dumps({"status": receipt["status"], "disposition": receipt["disposition"]}))


if __name__ == "__main__":
    main()
