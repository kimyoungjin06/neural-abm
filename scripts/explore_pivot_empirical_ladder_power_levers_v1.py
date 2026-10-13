#!/usr/bin/env python3
"""Exploratory screen of three power levers for the N11a v3 analysis (not registered).

N11a v3 recorded `UNDERPOWERED_AT_THE_EMPIRICAL_SHAPE` (bca896d): at the
N11b shape the learning worlds rarely beat the anticipatory rival B7. This
screen compares three candidate levers on synthetic worlds only, to choose
what a registered v4 should test. It reads no empirical row, registers no
threshold, and its seeds are opened for exploration.

- A, the test. Besides v3's three-way held-out win (A0), a joint model of
  the anticipatory and learned terms is fitted on confirmatory episodes. The
  learned coefficient wins when its one-sided 5% bound, from an
  episode-clustered sandwich variance, is positive. A1 holds B7's term
  constant over the episode; A2 lets it ramp linearly with the quarter.
- B, the evidence scope. Agents and analyst learn from the field's earlier
  episodes as well as the current one.
- C, the sample. The G4 guard is lifted: entrants x1.707 and authors at risk
  x1.275, the S1 and N11b removal counts.

The v3 script is imported by hash and its mechanisms are reused. Only the
linear learner (B4) is fitted, the learner every v3 learning replicate chose.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
V3_SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_ladder_recovery_v3.py"
V3_SCRIPT_SHA256 = "344dbbbe33dd4214b0cfe62eea1f70b9e287b8e1937f19c79f3c5a700cbb1969"
V3_CONTRACT = ROOT / "experiments" / "configs" / "pivot_empirical_ladder_recovery_v3.yaml"
V3_CONTRACT_SHA256 = "28477cbc6003ab69a767c31c93eb9dba460f49f783687133635b963883fbe855"
EXPLORATION_SEED = 20261105
WORLDS = ("W1_learning", "W3_static", "W6_static_aligned_drift")
CONFIGS = {
    "base": {"history": False, "entrants_scale": 1.0, "at_risk_scale": 1.0},
    "B_field_history": {"history": True, "entrants_scale": 1.0, "at_risk_scale": 1.0},
    "C_no_g4_guard": {"history": False, "entrants_scale": (19_132 + 13_529) / 19_132,
                      "at_risk_scale": (42_812_639 + 11_755_110) / 42_812_639},
}
LEARNED = "B4_linear"
ANTICIPATORY = "B7_anticipatory"
MODELS = ("B1", "B2", "B2t", "B3", LEARNED, ANTICIPATORY)
NON_LEARNING = ("B1", "B2", "B2t", "B3", ANTICIPATORY)
ONE_SIDED_Z = 1.6448536269514722
ANTICIPATORY_COLUMN = 1 + 4  # attention, four features, then the anticipatory term


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v3():
    if _sha256(V3_SCRIPT) != V3_SCRIPT_SHA256:
        raise RuntimeError("the v3 N11a script drifted from its bound hash")
    spec = importlib.util.spec_from_file_location("n11a_v3", V3_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_v3()
v1 = v3.v1
QUARTERS, FEATURES, EVIDENCE_QUARTERS = v3.QUARTERS, v3.FEATURES, v3.EVIDENCE_QUARTERS


def load_contract() -> dict[str, Any]:
    if _sha256(V3_CONTRACT) != V3_CONTRACT_SHA256:
        raise RuntimeError("the v3 N11a contract drifted from its bound hash")
    return v3.load_contract(V3_CONTRACT)


def shapes_for(contract: Mapping[str, Any], config: Mapping[str, Any], profile: str) -> tuple[list[dict[str, int]], int]:
    shapes, discovery = v3.episode_shapes(contract, profile)
    scaled = [{"field": s["field"], "at_risk": int(round(s["at_risk"] * config["at_risk_scale"])),
               "entrants": int(round(s["entrants"] * config["entrants_scale"]))} for s in shapes]
    return scaled, discovery


# -- the synthetic episode, with optional field history ------------------------------------


def simulate_episode(world: Mapping[str, Any], shape: Mapping[str, int], contract: Mapping[str, Any],
                     seed_seq: np.random.SeedSequence, history: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:
    """v3's episode for worlds without recruitment; learning agents also see ``history`` from q0."""

    if "recruitment" in world["confounds"]:
        raise ValueError("the screen covers worlds without recruitment only")
    feature_rng, entry_rng, outcome_rng, observe_rng, sample_rng = (np.random.default_rng(s) for s in seed_seq.spawn(5))
    mechanisms = contract["mechanisms"]
    n = int(shape["at_risk"])
    z = v3.draw_features(feature_rng, n, contract["features"], int(contract["seeds"]["standardization_seed"]))
    success = mechanisms["success"]
    weight_mean = np.asarray(success["episode_weight_mean"], dtype=float)
    weights = feature_rng.normal(weight_mean, float(success["episode_weight_sd"]))
    entry_spec = mechanisms["entry"]
    beta = np.asarray(entry_spec["beta"], dtype=float)
    attention = float(entry_spec["attention_slope"]) * np.arange(QUARTERS) / (QUARTERS - 1)
    static = z @ beta
    a0 = np.log(float(shape["entrants"])) - np.log(np.exp(attention).sum()) - np.log(np.exp(static).sum())
    base = a0 + static
    drift_end = v1._drift(world, QUARTERS - 1, weights, weight_mean)
    drift_score = z @ drift_end if np.any(drift_end) else None
    confounds = set(world["confounds"])
    kappa = float(world["kappa"])
    follow = mechanisms["follow_up"]
    minimum = int(mechanisms["agent_learning"]["minimum_visible_entrants"])
    hist_x, hist_y = history if history is not None else (np.zeros((0, FEATURES)), np.zeros(0, dtype=int))

    entry = np.full(n, -1, dtype=np.int16)
    ids = np.zeros(0, dtype=np.int64)
    true_follow = np.zeros((0, v3.FOLLOW_QUARTERS), dtype=bool)
    for quarter in range(QUARTERS):
        at_risk = np.flatnonzero(entry == -1)
        selectivity = np.zeros(at_risk.size)
        if drift_score is not None:
            selectivity += drift_score[at_risk] * quarter / (QUARTERS - 1)
        if kappa > 0:
            visible = entry[ids] <= quarter - EVIDENCE_QUARTERS - 1
            ev_x = np.vstack([hist_x, z[ids[visible]]])
            ev_y = np.concatenate([hist_y, true_follow[visible, :EVIDENCE_QUARTERS].any(axis=1).astype(int)])
            term = v1.learned_term("linear", {"C": 1.0}, ev_x, ev_y, z[at_risk], 0, minimum)
            if term is not None:
                selectivity += kappa * term
        logit = base[at_risk] + attention[quarter]
        if np.any(selectivity):
            static_weight = np.exp(static[at_risk])
            logit += selectivity - np.log(np.dot(static_weight, np.exp(selectivity)) / static_weight.sum())
        entered = at_risk[entry_rng.random(at_risk.size) < expit(logit)]
        entry[entered] = quarter
        success_logit = float(success["c0"]) + z[entered] @ weights + (quarter / (QUARTERS - 1) if "growth" in confounds else 0.0)
        succeeded = outcome_rng.random(entered.size) < expit(success_logit)
        rate = np.where(succeeded, float(follow["success_rate_per_quarter"]), float(follow["failure_rate_per_quarter"]))
        ids = np.concatenate([ids, entered])
        true_follow = np.vstack([true_follow, outcome_rng.random((entered.size, v3.FOLLOW_QUARTERS)) < rate[:, None]])

    precision = mechanisms["date_precision"]
    entrant_entry = entry[ids].astype(int)
    precise_entry = observe_rng.random(ids.size) < float(precision["entry_precise_share"])
    calendar = entrant_entry[:, None] + 1 + np.arange(v3.FOLLOW_QUARTERS)[None, :]
    seen = true_follow & (observe_rng.random(true_follow.shape) < v1.linkage(calendar, "linkage" in confounds))
    precise_follow = observe_rng.random(true_follow.shape) < float(precision["follow_up_precise_share"])
    label, visible_from = v3.analyst_labels(entrant_entry, precise_entry, seen, precise_follow)

    sampling = contract["analysis"]["sampling"]
    probability = min(1.0, max(float(sampling["min_subcohort"]), float(sampling["entrants_multiple"]) * ids.size) / n)
    subcohort = sample_rng.random(n) < probability
    is_entrant = np.zeros(n, dtype=bool)
    is_entrant[ids] = True
    sampled = np.flatnonzero(subcohort | is_entrant)
    precise_all = np.zeros(n, dtype=bool)
    precise_all[ids] = precise_entry
    return {
        "x": z[sampled],
        "entrant": is_entrant[sampled],
        "entry": entry[sampled].astype(int),
        "recruited": np.zeros(sampled.size, dtype=bool),
        "precise": precise_all[sampled],
        "probability": probability,
        "attention": attention,
        "evidence_x": z[ids],
        "evidence_label": label,
        "evidence_visible_from": visible_from,
        "true_label": true_follow[:, :EVIDENCE_QUARTERS].any(axis=1).astype(int),
        "entrants": int(ids.size),
    }


def ladder_rows(episode: Mapping[str, Any], minimum: int, history: tuple[np.ndarray, np.ndarray] | None) -> dict[str, np.ndarray]:
    """v3's weighted panel rows with the linear learner only; ``history`` joins the analyst's evidence from q0."""

    x = episode["x"]
    last, event = v3.last_rows(episode, True)
    weight_of = np.where(episode["entrant"], 1.0, 1.0 / episode["probability"])
    hist_x, hist_y = history if history is not None else (np.zeros((0, FEATURES)), np.zeros(0, dtype=int))
    ev_x, ev_label, ev_visible = episode["evidence_x"], episode["evidence_label"], episode["evidence_visible_from"]
    columns: dict[str, list[np.ndarray]] = {key: [] for key in ("y", "w", "attention", "x", "year", "quarter", "has_ev", "rate",
                                                                LEARNED, ANTICIPATORY)}
    anticipatory = v1.learned_term("linear", {"C": 1.0}, ev_x, ev_label, x, 0, minimum)
    for quarter in range(QUARTERS):
        rows = np.flatnonzero(last >= quarter)
        weight = weight_of[rows]
        visible = ev_visible <= quarter
        pool_x = np.vstack([hist_x, ev_x[visible]])
        pool_y = np.concatenate([hist_y, ev_label[visible]])
        has_evidence = int(pool_y.size) >= minimum
        columns["y"].append((event[rows] & (episode["entry"][rows] == quarter)).astype(float))
        columns["w"].append(weight)
        columns["attention"].append(np.full(rows.size, episode["attention"][quarter]))
        columns["x"].append(x[rows])
        columns["year"].append(np.full(rows.size, quarter // 4))
        columns["quarter"].append(np.full(rows.size, quarter / (QUARTERS - 1)))
        columns["has_ev"].append(np.full(rows.size, float(has_evidence)))
        columns["rate"].append(np.full(rows.size, float(pool_y.mean()) if has_evidence else 0.0))
        term = v1.learned_term("linear", {"C": 1.0}, pool_x, pool_y, x[rows], 0, minimum)
        columns[LEARNED].append(v3._centre(term, weight) if term is not None else np.zeros(rows.size))
        columns[ANTICIPATORY].append(v3._centre(anticipatory[rows], weight) if anticipatory is not None else np.zeros(rows.size))
    return {key: np.concatenate(values) for key, values in columns.items()}


# -- the tests ---------------------------------------------------------------------------


def joint_design(rows: Mapping[str, np.ndarray], ramp: bool) -> np.ndarray:
    parts = [rows["attention"][:, None], rows["x"], rows[ANTICIPATORY][:, None]]
    if ramp:
        parts.append((rows[ANTICIPATORY] * rows["quarter"])[:, None])
    parts.append(rows[LEARNED][:, None])
    return np.hstack(parts)


def clustered_test(x: np.ndarray, rows: Mapping[str, np.ndarray], anticipatory_column: int | None = None) -> dict[str, Any]:
    """Weighted fit; the last coefficient's one-sided 5% bound from an episode-clustered sandwich."""

    beta = v3.fit_logit(x, rows["y"], rows["w"])
    a = v3._with_intercept(x)
    p = expit(a @ beta)
    w = rows["w"]
    bread = (a * (w * p * (1 - p))[:, None]).T @ a
    _, cluster = np.unique(rows["episode"], return_inverse=True)
    groups = int(cluster.max()) + 1
    contributions = (w * (rows["y"] - p))[:, None] * a
    sums = np.zeros((groups, a.shape[1]))
    np.add.at(sums, cluster, contributions)
    inverse = np.linalg.pinv(bread)
    covariance = inverse @ (sums.T @ sums) @ inverse * groups / (groups - 1)
    se = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    coefficient = float(beta[-1])
    lower = coefficient - ONE_SIDED_Z * se
    return {"coefficient": coefficient, "se": se, "lower_bound": lower, "won": bool(lower > 0),
            "anticipatory_coefficient": None if anticipatory_column is None else float(beta[1 + anticipatory_column])}


def three_way(rows: Sequence[Mapping[str, np.ndarray]], discovery: int, count: int, bootstrap: Mapping[str, Any],
              rng: np.random.Generator) -> dict[str, Any]:
    """v3's registered rule with the linear learner as the only NABM model."""

    disc, conf = v1._concat(rows[:discovery]), v1._concat(rows[discovery:])
    fitted = {model: v3.fit_logit(v3.design(model, disc), disc["y"], disc["w"]) for model in MODELS}
    disc_ll = {model: float(v3.episode_loglik(fitted[model], model, disc, count).sum()) for model in MODELS}
    conf_ll = {model: v3.episode_loglik(fitted[model], model, conf, count)[discovery:] for model in MODELS}
    best_other = max(NON_LEARNING, key=lambda m: (disc_ll[m], m))
    comparisons = {}
    for name, rival in (("best_non_learning", best_other), ("b2", "B2"), ("anticipatory", ANTICIPATORY)):
        difference = conf_ll[LEARNED] - conf_ll[rival]
        lower = v1._bootstrap_lower(difference, int(bootstrap["resamples"]), float(bootstrap["lower_quantile"]), rng)
        comparisons[name] = {"rival": rival, "mean": float(difference.mean()), "lower_bound": lower,
                             "won": bool(difference.mean() > 0 and lower > 0)}
    return {"won": all(c["won"] for c in comparisons.values()), "best_non_learning": best_other, "comparisons": comparisons,
            "learned_coefficient": float(fitted[LEARNED][-1]), "anticipatory_coefficient": float(fitted[ANTICIPATORY][-1])}


# -- tasks -------------------------------------------------------------------------------


def run_task(contract: Mapping[str, Any], profile: str, config_name: str, world_id: str, replicate: int) -> dict[str, Any]:
    started = time.perf_counter()
    config = CONFIGS[config_name]
    world_index = [w["id"] for w in contract["worlds"]].index(world_id)
    world = contract["worlds"][world_index]
    shapes, discovery = shapes_for(contract, config, profile)
    count = len(shapes)
    minimum = int(contract["analysis"]["evidence"]["minimum_visible_entrants"])
    base = np.random.SeedSequence((EXPLORATION_SEED, list(CONFIGS).index(config_name), world_index, replicate))
    seeds = base.spawn(count + 1)
    agent_history: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    analyst_history: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    rows = []
    with threadpool_limits(limits=1):
        for index, (shape, seq) in enumerate(zip(shapes, seeds[:count], strict=True)):
            field = shape["field"]
            agent_pool = analyst_pool = None
            if config["history"] and agent_history.get(field):
                agent_pool = (np.vstack([h[0] for h in agent_history[field]]), np.concatenate([h[1] for h in agent_history[field]]))
                analyst_pool = (np.vstack([h[0] for h in analyst_history[field]]), np.concatenate([h[1] for h in analyst_history[field]]))
            episode = simulate_episode(world, shape, contract, seq, agent_pool)
            part = ladder_rows(episode, minimum, analyst_pool)
            part["episode"] = np.full(part["y"].size, index)
            rows.append(part)
            agent_history.setdefault(field, []).append((episode["evidence_x"], episode["true_label"]))
            analyst_history.setdefault(field, []).append((episode["evidence_x"], episode["evidence_label"]))
        rng = np.random.default_rng(seeds[count])
        a0 = three_way(rows, discovery, count, contract["decision"]["bootstrap"], rng)
        conf = v1._concat(rows[discovery:])
        every = v1._concat(rows)
        tests = {
            "A0_three_way": a0,
            "A1_joint_confirmatory": clustered_test(joint_design(conf, False), conf, ANTICIPATORY_COLUMN),
            "A2_joint_ramp_confirmatory": clustered_test(joint_design(conf, True), conf, ANTICIPATORY_COLUMN),
            "A1_joint_all_episodes": clustered_test(joint_design(every, False), every, ANTICIPATORY_COLUMN),
            "A2_joint_ramp_all_episodes": clustered_test(joint_design(every, True), every, ANTICIPATORY_COLUMN),
        }
    return {
        "profile": profile,
        "config": config_name,
        "world": world_id,
        "kappa": world["kappa"],
        "replicate": replicate,
        "tests": tests,
        "events": int(sum(part["y"].sum() for part in rows)),
        "confirmatory_events": int(conf["y"].sum()),
        "panel_rows": int(sum(part["y"].size for part in rows)),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_task(*args)


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for config in CONFIGS:
        for world in WORLDS:
            selected = [r for r in results if r["config"] == config and r["world"] == world]
            if not selected:
                continue
            entry = {"replicates": len(selected), "median_events": float(np.median([r["events"] for r in selected]))}
            for test in selected[0]["tests"]:
                outcomes = [r["tests"][test] for r in selected]
                entry[test] = {
                    "wins": int(sum(o["won"] for o in outcomes)),
                    "median_learned_coefficient": float(np.median([o.get("coefficient", o.get("learned_coefficient")) for o in outcomes])),
                    "median_anticipatory_coefficient": float(np.median([o["anticipatory_coefficient"] for o in outcomes])),
                }
                if test != "A0_three_way":
                    entry[test]["median_se"] = float(np.median([o["se"] for o in outcomes]))
                else:
                    entry[test]["beats_b2"] = int(sum(o["comparisons"]["b2"]["won"] for o in outcomes))
                    entry[test]["beats_anticipatory"] = int(sum(o["comparisons"]["anticipatory"]["won"] for o in outcomes))
            table[f"{config}/{world}"] = entry
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS))
    args = parser.parse_args()
    contract = load_contract()
    tasks = [(contract, args.profile, c, w, r) for c in args.configs for w in args.worlds for r in range(args.replicates)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.workers <= 1:
        results = [_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_task, tasks))
    (args.output_dir / "results.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in results), encoding="utf-8")
    summary = {"status": "EXPLORATORY_NOT_REGISTERED", "script_sha256": _sha256(Path(__file__)),
               "v3_script_sha256": V3_SCRIPT_SHA256, "v3_contract_sha256": V3_CONTRACT_SHA256,
               "seed": EXPLORATION_SEED, "profile": args.profile, "replicates": args.replicates, "empirical_rows_read": 0, "table": summarize(results)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["table"], indent=1))


if __name__ == "__main__":
    main()
