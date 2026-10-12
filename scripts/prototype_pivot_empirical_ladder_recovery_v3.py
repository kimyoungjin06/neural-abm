#!/usr/bin/env python3
"""N11a v3: does the model ladder still recover learning at the empirical shape?

v1 and v2 passed identification in worlds of 600 authors at risk per episode,
a third of whom entered. N11b measured the empirical shape: about 680,000
authors at risk per confirmatory episode, about 3 entries per 100,000 per
quarter, 42.5% of entry dates known only to the year, sparse count features,
and an evidence pool that is small and appears late. v3 reruns the recovery
test at that shape with the analysis N11c and N11d would apply to data:

- a case-cohort sample with inverse-probability weights;
- imprecise entrants censored from the start of their entry year;
- evidence labels built only from what dated records show;
- entries recruited by an earlier entrant's follow-up censored as a
  competing risk.

The v1 learners, drift, linkage, bootstrap, and precommit check are reused
unchanged, verified by hash. No empirical row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml
from scipy.special import expit, ndtr, ndtri
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_ladder_recovery_v3.yaml"
V1_SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_ladder_recovery_v1.py"
V1_SCRIPT_SHA256 = "3ac85cf8a55c8b8ec695c8543c0cf1a6ad114b0f5b42607dcde73decaa415102"
ANTICIPATORY = "B7_anticipatory"
FOLLOW_QUARTERS = 24
POOL_QUARTERS = (5, 8, 11)


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
QUARTERS = v1.QUARTERS
FEATURES = v1.FEATURES
EVIDENCE_QUARTERS = v1.EVIDENCE_QUARTERS
LEARNERS = v1.LEARNERS
MODELS = (*v1.MODELS, ANTICIPATORY)
NABM = v1.NABM
NON_LEARNING = (*v1.NON_LEARNING, ANTICIPATORY)
CONFOUNDS = {"growth", "linkage", "independent_drift", "aligned_drift", "recruitment"}


# -- contract ----------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("the N11a v3 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_empirical_ladder_recovery.v3":
        raise ValueError("unexpected N11a v3 schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("the N11a v3 contract is not a precommit design")
    amends = contract["amends"]
    for key in ("contract_v2", "script_v2", "result_v2"):
        if _sha256(ROOT / amends[key]["path"]) != amends[key]["sha256"]:
            raise ValueError(f"the amended {key} drifted from its bound hash")
    v2 = yaml.safe_load((ROOT / amends["contract_v2"]["path"]).read_text(encoding="utf-8"))
    for key in ("success", "follow_up", "growth_confound", "linkage_confound", "linkage_without_confound",
                "independent_drift", "aligned_drift"):
        if contract["mechanisms"][key] != v2["mechanisms"][key]:
            raise ValueError(f"v3 may not change the v2 {key} mechanism")
    for key in ("learners", "shared_inputs", "nabm"):
        if contract["ladder"][key] != v2["ladder"][key]:
            raise ValueError(f"v3 may not change the v2 ladder {key}")
    if tuple(contract["ladder"]["models"]) != MODELS or tuple(contract["ladder"]["non_learning"]) != NON_LEARNING:
        raise ValueError("the v3 ladder must be v2's")
    for key in ("bootstrap", "nabm_win", "replicates", "must_recover_at_least", "must_not_false_recover_at_most"):
        if contract["decision"][key] != v2["decision"][key]:
            raise ValueError(f"v3 may not change the v2 decision {key}")
    if contract["episode_shape"]["quarters"] != QUARTERS or len(contract["features"]["order"]) != FEATURES:
        raise ValueError("the episode structure changed")
    if contract["features"]["order"] != v2["episode_structure"]["features"]:
        raise ValueError("the feature order must stay v2's")
    for part in ("discovery", "confirmatory"):
        for field, at_risk, entrants in contract["episode_shape"][part]:
            if not (0 < int(entrants) < int(at_risk)):
                raise ValueError("an episode shape needs entrants between zero and the risk set")
    roles = {w["id"]: w["role"] for w in contract["worlds"]}
    if sorted(r for r in roles.values() if r == "MUST_RECOVER") != ["MUST_RECOVER"] * 2:
        raise ValueError("two worlds must be registered as must-recover")
    for world_id in ("W3_static", "W4_static_confounded", "W5_static_independent_drift", "W6_static_aligned_drift", "W8_static_recruitment"):
        if roles.get(world_id) != "MUST_NOT_FALSE_RECOVER":
            raise ValueError(f"{world_id} must be must-not-falsely-recover")
    for world in contract["worlds"]:
        if not set(world["confounds"]) <= CONFOUNDS:
            raise ValueError(f"unknown confound in {world['id']}")
        if world["role"] == "MUST_NOT_FALSE_RECOVER" and world["kappa"] != 0:
            raise ValueError("a must-not-falsely-recover world must not learn")
        if world["role"] == "MUST_RECOVER" and world["kappa"] <= 0:
            raise ValueError("a must-recover world must learn")
    seeds = contract["seeds"]
    if int(seeds["base_seed"]) in {int(s) for s in seeds["opened"]} or int(seeds["standardization_seed"]) in {int(s) for s in seeds["opened"]}:
        raise ValueError("v3 must use fresh seeds")
    v1.validate_contract(yaml.safe_load((ROOT / "experiments" / "configs" / "pivot_empirical_ladder_recovery_v1.yaml").read_text(encoding="utf-8")))


def episode_shapes(contract: Mapping[str, Any], profile: str) -> tuple[list[dict[str, int]], int]:
    """The profile's episodes, discovery first, and the number of discovery episodes."""

    settings = contract["profiles"][profile]
    scale = float(settings["at_risk_scale"])
    cap = float(settings["entrants_cap_share"])
    shapes: list[dict[str, int]] = []
    counts = {}
    for part in ("discovery", "confirmatory"):
        rows = contract["episode_shape"][part]
        chosen = range(len(rows)) if settings[f"{part}_episodes"] == "ALL" else [int(i) for i in settings[f"{part}_episodes"]]
        for index in chosen:
            field, at_risk, entrants = (int(v) for v in rows[index])
            n = max(int(round(at_risk * scale)), 100)
            shapes.append({"field": field, "at_risk": n, "entrants": max(1, min(entrants, int(cap * n)))})
        counts[part] = len(chosen)
    return shapes, counts["discovery"]


# -- features ----------------------------------------------------------------------------


def _counts(rng: np.random.Generator, n: int, spec: Mapping[str, Any]) -> np.ndarray:
    """Productivity, career age, subfield share, and incumbent coauthors from a Gaussian copula."""

    rho = float(spec["copula_correlation"])
    chol = np.linalg.cholesky((1 - rho) * np.eye(FEATURES) + rho * np.ones((FEATURES, FEATURES)))
    g = rng.standard_normal((n, FEATURES)) @ chol.T
    p = spec["marginal_parameters"]
    productivity = np.maximum(1.0, np.rint(np.exp(p["productivity"]["mu"] + p["productivity"]["sigma"] * g[:, 0])))
    career = np.clip(1.0 + np.rint(np.exp(p["career_age"]["mu"] + p["career_age"]["sigma"] * g[:, 1])), 1.0, float(p["career_age"]["max"]))
    u = ndtr(g[:, 2])
    zero = float(p["proximity"]["zero_share"])
    proximity = np.where(u < zero, 0.0, np.clip((u - zero) / (1 - zero), 0.0, 1.0) ** float(p["proximity"]["power"]))
    u = ndtr(g[:, 3])
    zero = float(p["network"]["zero_share"])
    tail = ndtri(np.clip((u - zero) / (1 - zero), 1e-12, 1 - 1e-12))
    network = np.where(u < zero, 0.0, np.maximum(1.0, np.rint(np.exp(float(p["network"]["mu"]) + tail))))
    return np.column_stack([productivity, career, proximity, network])


def _transform(counts: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log1p(counts[:, 0]), np.log1p(counts[:, 1]), counts[:, 2], np.log1p(counts[:, 3])])


@lru_cache(maxsize=4)
def _standardization(spec_json: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    reference = _transform(_counts(np.random.default_rng(seed), 1_000_000, json.loads(spec_json)))
    return reference.mean(axis=0), reference.std(axis=0)


def draw_features(rng: np.random.Generator, n: int, spec: Mapping[str, Any], seed: int) -> np.ndarray:
    mean, sd = _standardization(json.dumps(spec, sort_keys=True), seed)
    return (_transform(_counts(rng, n, spec)) - mean) / sd


# -- the synthetic episode ---------------------------------------------------------------


def analyst_labels(entry: np.ndarray, precise_entry: np.ndarray, seen: np.ndarray, precise_follow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each entrant's analyst label and the quarter from which it is visible.

    A precise entrant's label is a seen follow-up in the four quarters after
    entry, dated to the quarter or to a year lying wholly in that window; it
    is visible five quarters after entry. An imprecise entrant's label is a
    seen follow-up dated in the year after the entry year; it is visible from
    the first quarter two years after the entry year. Column k of the
    follow-up arrays is the quarter entry + 1 + k.
    """

    window = slice(0, EVIDENCE_QUARTERS)
    quarter_dated = (seen[:, window] & precise_follow[:, window]).any(axis=1)
    whole_year = (entry % 4 == 3) & (seen[:, window] & ~precise_follow[:, window]).any(axis=1)
    precise_label = quarter_dated | whole_year
    start = 4 * (entry // 4 + 1) - entry - 1
    columns = start[:, None] + np.arange(4)[None, :]
    next_year = np.take_along_axis(seen, columns, axis=1).any(axis=1)
    label = np.where(precise_entry, precise_label, next_year)
    visible_from = np.where(precise_entry, entry + EVIDENCE_QUARTERS + 1, 4 * (entry // 4) + 8)
    return label.astype(int), visible_from


def simulate_episode(world: Mapping[str, Any], shape: Mapping[str, int], contract: Mapping[str, Any], seed_seq: np.random.SeedSequence) -> dict[str, Any]:
    """One episode over the full risk set, returned as its case-cohort sample."""

    feature_rng, entry_rng, outcome_rng, observe_rng, sample_rng = (np.random.default_rng(s) for s in seed_seq.spawn(5))
    mechanisms = contract["mechanisms"]
    n = int(shape["at_risk"])
    z = draw_features(feature_rng, n, contract["features"], int(contract["seeds"]["standardization_seed"]))
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
    agents = mechanisms["agent_learning"]
    minimum = int(agents["minimum_visible_entrants"])
    recruitment = mechanisms["recruitment"]
    rho, candidates, bandwidth = float(recruitment["rho"]), int(recruitment["candidates"]), float(recruitment["bandwidth_h"])

    entry = np.full(n, -1, dtype=np.int16)
    recruited = np.zeros(n, dtype=bool)
    ids = np.zeros(0, dtype=np.int64)
    true_follow = np.zeros((0, FOLLOW_QUARTERS), dtype=bool)
    for quarter in range(QUARTERS):
        at_risk = np.flatnonzero(entry == -1)
        selectivity = np.zeros(at_risk.size)
        if drift_score is not None:
            selectivity += drift_score[at_risk] * quarter / (QUARTERS - 1)
        if kappa > 0:
            visible = entry[ids] <= quarter - EVIDENCE_QUARTERS - 1
            term = v1.learned_term("linear", {"C": 1.0}, z[ids[visible]], true_follow[visible, :EVIDENCE_QUARTERS].any(axis=1).astype(int),
                                   z[at_risk], 0, minimum)
            if term is not None:
                selectivity += kappa * term
        logit = base[at_risk] + attention[quarter]
        if np.any(selectivity):
            static_weight = np.exp(static[at_risk])
            logit += selectivity - np.log(np.dot(static_weight, np.exp(selectivity)) / static_weight.sum())
        entered = at_risk[entry_rng.random(at_risk.size) < expit(logit)]
        entry[entered] = quarter
        recruits = []
        if "recruitment" in confounds and ids.size:
            lag = quarter - entry[ids].astype(int) - 1
            active = np.flatnonzero((lag >= 0) & (lag < FOLLOW_QUARTERS))
            for recruiter in active[true_follow[active, lag[active]]]:
                if entry_rng.random() >= rho:
                    continue
                pool = at_risk[entry_rng.integers(0, at_risk.size, size=candidates)]
                distance = ((z[pool] - z[ids[recruiter]]) ** 2).sum(axis=1)
                weight = np.exp(-distance / (2 * bandwidth**2))
                pick = int(pool[entry_rng.choice(pool.size, p=weight / weight.sum())])
                if entry[pick] == -1:
                    entry[pick] = quarter
                    recruited[pick] = True
                    recruits.append(pick)
        new = np.concatenate([entered, np.asarray(recruits, dtype=np.int64)])
        success_logit = float(success["c0"]) + z[new] @ weights + (quarter / (QUARTERS - 1) if "growth" in confounds else 0.0)
        succeeded = outcome_rng.random(new.size) < expit(success_logit)
        rate = np.where(succeeded, float(follow["success_rate_per_quarter"]), float(follow["failure_rate_per_quarter"]))
        ids = np.concatenate([ids, new])
        true_follow = np.vstack([true_follow, outcome_rng.random((new.size, FOLLOW_QUARTERS)) < rate[:, None]])

    precision = mechanisms["date_precision"]
    entrant_entry = entry[ids].astype(int)
    precise_entry = observe_rng.random(ids.size) < float(precision["entry_precise_share"])
    calendar = entrant_entry[:, None] + 1 + np.arange(FOLLOW_QUARTERS)[None, :]
    seen = true_follow & (observe_rng.random(true_follow.shape) < v1.linkage(calendar, "linkage" in confounds))
    precise_follow = observe_rng.random(true_follow.shape) < float(precision["follow_up_precise_share"])
    label, visible_from = analyst_labels(entrant_entry, precise_entry, seen, precise_follow)

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
        "recruited": recruited[sampled],
        "precise": precise_all[sampled],
        "in_subcohort": subcohort[sampled],
        "probability": probability,
        "attention": attention,
        "evidence_x": z[ids],
        "evidence_label": label,
        "evidence_visible_from": visible_from,
        "entrants": int(ids.size),
        "precise_entrants": int(precise_entry.sum()),
        "recruited_entrants": int(recruited[ids].sum()),
        "at_risk": n,
    }


# -- the analysis --------------------------------------------------------------------------


def last_rows(episode: Mapping[str, Any], censor_recruited: bool) -> tuple[np.ndarray, np.ndarray]:
    """Each sampled author's last panel quarter and whether that row is an entry event."""

    entry, entrant, precise, recruited = episode["entry"], episode["entrant"], episode["precise"], episode["recruited"]
    last = np.full(entry.size, QUARTERS - 1)
    event = entrant & precise & ~(recruited & censor_recruited)
    last = np.where(event, entry, last)
    last = np.where(entrant & precise & recruited & censor_recruited, entry - 1, last)
    last = np.where(entrant & ~precise, 4 * (entry // 4) - 1, last)
    return last, event


def _centre(term: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return term - float(np.dot(term, weight) / weight.sum())


def ladder_rows(episode: Mapping[str, Any], ladder: Mapping[str, Any], minimum: int, seed_seq: np.random.SeedSequence,
                *, censor_recruited: bool = True) -> dict[str, np.ndarray]:
    """One weighted Bernoulli row per sampled author and quarter at risk, with every model's inputs."""

    x = episode["x"]
    last, event = last_rows(episode, censor_recruited)
    weight_of = np.where(episode["entrant"], 1.0, 1.0 / episode["probability"])
    ev_x, ev_label, ev_visible = episode["evidence_x"], episode["evidence_label"], episode["evidence_visible_from"]
    seeds = {name: np.random.default_rng(s) for name, s in zip(LEARNERS, seed_seq.spawn(len(LEARNERS)), strict=True)}
    columns: dict[str, list[np.ndarray]] = {key: [] for key in ("y", "w", "attention", "x", "year", "has_ev", "rate", *LEARNERS, ANTICIPATORY)}
    anticipatory = v1.learned_term("linear", {"C": 1.0}, ev_x, ev_label, x, 0, minimum)
    for quarter in range(QUARTERS):
        rows = np.flatnonzero(last >= quarter)
        weight = weight_of[rows]
        visible = ev_visible <= quarter
        has_evidence = int(visible.sum()) >= minimum
        columns["y"].append((event[rows] & (episode["entry"][rows] == quarter)).astype(float))
        columns["w"].append(weight)
        columns["attention"].append(np.full(rows.size, episode["attention"][quarter]))
        columns["x"].append(x[rows])
        columns["year"].append(np.full(rows.size, quarter // 4))
        columns["has_ev"].append(np.full(rows.size, float(has_evidence)))
        columns["rate"].append(np.full(rows.size, float(ev_label[visible].mean()) if has_evidence else 0.0))
        for model, family in LEARNERS.items():
            term = v1.learned_term(family, ladder["learners"][family], ev_x[visible], ev_label[visible], x[rows],
                                   int(seeds[model].integers(2**31)), minimum)
            columns[model].append(_centre(term, weight) if term is not None else np.zeros(rows.size))
        columns[ANTICIPATORY].append(_centre(anticipatory[rows], weight) if anticipatory is not None else np.zeros(rows.size))
    return {key: np.concatenate(values) for key, values in columns.items()}


def design(model: str, rows: Mapping[str, np.ndarray]) -> np.ndarray:
    if model == ANTICIPATORY:
        return np.hstack([rows["attention"][:, None], rows["x"], rows[ANTICIPATORY][:, None]])
    return v1.design(model, rows)


def _with_intercept(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), x])


def _weighted_loglik(a: np.ndarray, y: np.ndarray, w: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = a @ beta
    return w * (y * eta - np.logaddexp(0.0, eta))


def fit_logit(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Weighted unpenalized maximum likelihood by Newton steps with step halving; the intercept comes first."""

    a = _with_intercept(x)
    beta = np.zeros(a.shape[1])
    beta[0] = np.log(np.dot(w, y) / np.dot(w, 1 - y))
    ll = float(_weighted_loglik(a, y, w, beta).sum())
    for _ in range(max_iter):
        p = expit(a @ beta)
        gradient = a.T @ (w * (y - p))
        hessian = (a * (w * p * (1 - p))[:, None]).T @ a
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        scale = 1.0
        while True:
            candidate = beta + scale * step
            new = float(_weighted_loglik(a, y, w, candidate).sum())
            if new >= ll or scale < 1e-6:
                break
            scale /= 2
        if new < ll:
            break
        change = float(np.max(np.abs(candidate - beta)))
        beta, ll = candidate, new
        if change < tol:
            break
    return beta


def fit(model: str, rows: Mapping[str, np.ndarray]) -> np.ndarray:
    return fit_logit(design(model, rows), rows["y"], rows["w"])


def episode_loglik(beta: np.ndarray, model: str, rows: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    ll = _weighted_loglik(_with_intercept(design(model, rows)), rows["y"], rows["w"], beta)
    return np.bincount(rows["episode"], weights=ll, minlength=count)


def score(rows: Sequence[Mapping[str, np.ndarray]], discovery: int, count: int, decision: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Fit on discovery, choose by discovery likelihood, and score the three-way win on confirmatory episodes."""

    disc = v1._concat(rows[:discovery])
    conf = v1._concat(rows[discovery:])
    fitted = {model: fit(model, disc) for model in MODELS}
    disc_ll = {model: float(episode_loglik(fitted[model], model, disc, count).sum()) for model in MODELS}
    conf_ll = {model: episode_loglik(fitted[model], model, conf, count)[discovery:] for model in MODELS}
    best_nabm = max(NABM, key=lambda m: (disc_ll[m], m))
    best_other = max(NON_LEARNING, key=lambda m: (disc_ll[m], m))
    bootstrap = decision["bootstrap"]
    comparisons = {}
    for name, rival in (("best_non_learning", best_other), ("b2", "B2"), ("anticipatory", ANTICIPATORY)):
        difference = conf_ll[best_nabm] - conf_ll[rival]
        lower = v1._bootstrap_lower(difference, int(bootstrap["resamples"]), float(bootstrap["lower_quantile"]), rng)
        comparisons[name] = {"rival": rival, "mean": float(difference.mean()), "median": float(np.median(difference)),
                             "lower_bound": lower, "won": bool(difference.mean() > 0 and lower > 0)}
    return {
        "best_nabm": best_nabm,
        "best_non_learning": best_other,
        "comparisons": comparisons,
        "nabm_win": all(c["won"] for c in comparisons.values()),
        "learned_coefficient": float(fitted[best_nabm][-1]),
        "anticipatory_coefficient": float(fitted[ANTICIPATORY][-1]),
        "discovery_ll": disc_ll,
        "confirmatory_ll": {model: float(values.sum()) for model, values in conf_ll.items()},
    }


def _evidence_pool(episodes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {f"quarter_{q}": float(np.median([int((e["evidence_visible_from"] <= q).sum()) for e in episodes])) for q in POOL_QUARTERS}


def run_task(contract: Mapping[str, Any], profile: str, world_index: int, replicate: int) -> dict[str, Any]:
    world = contract["worlds"][world_index]
    shapes, discovery = episode_shapes(contract, profile)
    count = len(shapes)
    minimum = int(contract["analysis"]["evidence"]["minimum_visible_entrants"])
    base = np.random.SeedSequence((int(contract["seeds"]["base_seed"]), world_index, replicate))
    episode_seeds = base.spawn(count + 1)
    rows, censored_free = [], []
    episodes = []
    with threadpool_limits(limits=1):
        for index, (shape, seq) in enumerate(zip(shapes, episode_seeds[:count], strict=True)):
            simulate_seed, ladder_seed = seq.spawn(2)
            episode = simulate_episode(world, shape, contract, simulate_seed)
            part = ladder_rows(episode, contract["ladder"], minimum, ladder_seed)
            part["episode"] = np.full(part["y"].size, index)
            rows.append(part)
            if "recruitment" in world["confounds"] and world["role"] == "MUST_NOT_FALSE_RECOVER":
                free = ladder_rows(episode, contract["ladder"], minimum, ladder_seed, censor_recruited=False)
                free["episode"] = np.full(free["y"].size, index)
                censored_free.append(free)
            episodes.append({key: episode[key] for key in ("evidence_visible_from", "entrants", "precise_entrants", "recruited_entrants")})
        rng = np.random.default_rng(episode_seeds[count])
        result = score(rows, discovery, count, contract["decision"], rng)
        uncensored = score(censored_free, discovery, count, contract["decision"], rng) if censored_free else None
    entrants = np.asarray([e["entrants"] for e in episodes])
    return {
        "world": world["id"],
        "role": world["role"],
        "replicate": replicate,
        **result,
        "without_recruitment_censoring": None if uncensored is None else {
            "nabm_win": uncensored["nabm_win"], "best_nabm": uncensored["best_nabm"],
            "comparisons": uncensored["comparisons"], "learned_coefficient": uncensored["learned_coefficient"]},
        "entrants_per_episode_median": float(np.median(entrants)),
        "precise_share": float(sum(e["precise_entrants"] for e in episodes) / max(int(entrants.sum()), 1)),
        "recruited_share": float(sum(e["recruited_entrants"] for e in episodes) / max(int(entrants.sum()), 1)),
        "confirmatory_evidence_pool_median": _evidence_pool(episodes[discovery:]),
        "panel_rows": int(sum(part["y"].size for part in rows)),
        "events": int(sum(part["y"].sum() for part in rows)),
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
        median = (lambda values: float(np.median(values)) if selected else None)
        worlds[world["id"]] = {
            "role": world["role"],
            "kappa": world["kappa"],
            "nabm_wins": wins,
            "replicates": len(selected),
            "meets_role": ok,
            "beats": {name: sum(r["comparisons"][name]["won"] for r in selected) for name in ("best_non_learning", "b2", "anticipatory")},
            "median_mean_ll_difference": {name: median([r["comparisons"][name]["mean"] for r in selected])
                                          for name in ("best_non_learning", "b2", "anticipatory")},
            "best_nabm": sorted({r["best_nabm"] for r in selected}),
            "best_non_learning": sorted({r["best_non_learning"] for r in selected}),
            "median_learned_coefficient": median([r["learned_coefficient"] for r in selected]),
            "median_anticipatory_coefficient": median([r["anticipatory_coefficient"] for r in selected]),
            "median_entrants_per_episode": median([r["entrants_per_episode_median"] for r in selected]),
            "median_precise_share": median([r["precise_share"] for r in selected]),
            "median_recruited_share": median([r["recruited_share"] for r in selected]),
            "median_confirmatory_evidence_pool": {q: median([r["confirmatory_evidence_pool_median"][q] for r in selected])
                                                  for q in (f"quarter_{k}" for k in POOL_QUARTERS)},
            "wins_without_recruitment_censoring": (sum(r["without_recruitment_censoring"]["nabm_win"] for r in selected)
                                                   if selected and selected[0]["without_recruitment_censoring"] is not None else None),
        }
    must_not = [w for w in worlds.values() if w["role"] == "MUST_NOT_FALSE_RECOVER"]
    must = [w for w in worlds.values() if w["role"] == "MUST_RECOVER"]
    if not all(w["meets_role"] for w in must_not):
        disposition = "FALSE_RECOVERY_AT_THE_EMPIRICAL_SHAPE"
    elif not all(w["meets_role"] for w in must):
        disposition = "UNDERPOWERED_AT_THE_EMPIRICAL_SHAPE"
    else:
        disposition = "RECOVERY_PASSES_AT_THE_EMPIRICAL_SHAPE"
    return {"disposition": disposition, "action": rule["actions"][disposition], "worlds": worlds}


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot empirical ladder recovery v3 (N11a v3)",
        "",
        f"Profile: `{summary['profile']}`. Disposition: `{summary['decision']['disposition']}`",
        "",
        "| World | Role | κ | NABM wins | Beats best non-learning | Beats B2 | Beats B7 | Learned coefficient | Recruited share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for world_id, w in summary["decision"]["worlds"].items():
        coefficient = w["median_learned_coefficient"]
        lines.append(f"| {world_id} | {w['role']} | {w['kappa']} | {w['nabm_wins']}/{w['replicates']} | {w['beats']['best_non_learning']} | "
                     f"{w['beats']['b2']} | {w['beats']['anticipatory']} | {coefficient:.2f} | {w['median_recruited_share']:.3f} |")
    lines += ["", "Synthetic worlds at the N11b shape only. No empirical row was read.", ""]
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
        "schema_version": "pivot_empirical_ladder_recovery_receipt.v3",
        "status": "COMPLETE",
        "disposition": summary["decision"]["disposition"],
        "contract_sha256": _sha256(contract_path),
        "script_sha256": _sha256(Path(__file__)),
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
        shapes, discovery = episode_shapes(contract, args.profile)
        print(json.dumps({"status": "PRECOMMIT_PLAN_VALID", "contract_sha256": _sha256(config_path),
                          "worlds": [w["id"] for w in contract["worlds"]], "episodes": len(shapes), "discovery_episodes": discovery,
                          "authors_at_risk": sum(s["at_risk"] for s in shapes), "source_bindings": v1.verify_source_bindings(contract)},
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
