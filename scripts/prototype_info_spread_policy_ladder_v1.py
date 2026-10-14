"""Information-spread policy ladder, stage D1 prototype (repository-only).

Design: docs/research/info-spread-policy-ladder-design-v1.md. Nothing here is
frozen or registered. Stage D1 builds the world and the non-LLM arms, checks
the nesting rules exactly, and measures feasibility only: revealed outcomes
per agent, runtime per agent-step, and the archetype query budget. The
feasibility report carries no behavioral outcome of any arm.

World. Agents sit on a Watts-Strogatz graph. Each step, new items enter, each
shared first by a few seed agents. An item has four features in [0, 1] and a
latent veracity drawn from a geometry family. An agent who has not shared an
item decides whether to share it each time a neighbor newly shares it, as in
Watts-Granovetter threshold dynamics; a share is absorbing. Its policy gives
the probability that the item is true, and it shares when that probability
reaches c/(1 + c), where c is its cost of sharing a false item. A share that is later fact-checked
becomes an outcome record: visible `delay` steps after the item entered, for
a `visibility` share of items.

The network, thresholds, and costs belong to the world seed; the item stream
belongs to the replicate. Donor runs for the precomputed-table arms therefore
see the same agents and a different stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402


ITEM_FEATURES = ("credibility", "novelty", "intensity", "fit")
EXPOSURE_FEATURES = ("exposure_share", "log_exposures")
TRUST_FEATURE = "sharer_trust"
GEOMETRIES = ("linear", "smooth", "axis", "null")
REGIMES = ("A", "B")

# Stream tags for keyed randomness.
TAG_GRAPH, TAG_AGENTS, TAG_ITEMS, TAG_CALIBRATION, TAG_EXPLORE, TAG_SHIFT = 1, 2, 3, 4, 5, 6
EARLY_SHOCK_STEPS = 10


@dataclass(frozen=True)
class WorldConfig:
    agents: int = 1000
    degree: int = 6
    rewire: float = 0.1
    steps: int = 60
    items_per_step: int = 20
    seeds_per_item: int = 3
    delay: int = 4
    visibility: float = 0.5
    geometry: str = "linear"
    base_rate: float = 0.4
    regime: str = "A"
    cost_levels: tuple[float, ...] = (0.5, 1.0, 2.0)
    threshold_low: float = 0.1
    threshold_high: float = 0.4
    refit_every: int = 4
    epsilon: float = 0.0
    world_seed: int = 0
    # D2 positive controls: from a step drawn per replicate, the credibility
    # weight of the veracity relation becomes this value. None means no shift.
    shift_credibility_weight: float | None = None

    def feature_names(self) -> tuple[str, ...]:
        names = ITEM_FEATURES + EXPOSURE_FEATURES
        return names + (TRUST_FEATURE,) if self.regime == "B" else names


LEARNER_SETTINGS: dict[str, dict[str, Any]] = {
    "linear": {"learning_rate": 0.05, "epochs": 60, "weight_decay": 1e-3},
    "mlp": {"learning_rate": 0.05, "epochs": 60, "weight_decay": 1e-3, "hidden_width": 8},
    "tree": {"rounds": 6, "learning_rate": 0.3, "leaf_l2": 1.0, "max_depth": 3, "min_samples_leaf": 10},
}


# -- world -------------------------------------------------------------------


def _veracity_logit_without_intercept(
    geometry: str, z: np.ndarray, credibility_weight: float = 3.0
) -> np.ndarray:
    cred, nov, inten, fit = (z[:, k] for k in range(4))
    if geometry == "null":
        return np.zeros(len(z))
    linear = credibility_weight * (cred - 0.5) - 1.0 * (nov - 0.5) - 2.0 * (inten - 0.5)
    if geometry == "linear":
        return linear
    if geometry == "smooth":
        return linear + 8.0 * (cred - 0.5) * (fit - 0.5)
    if geometry == "axis":
        return (
            3.0 * (cred > 0.55) - 3.0 * (inten > 0.65) + 1.5 * (fit > 0.5)
        ).astype(np.float64)
    raise ValueError(f"unknown geometry: {geometry}")


def _calibrated_intercept(config: WorldConfig) -> float:
    """Intercept that gives the base rate on a keyed calibration sample."""

    rng = np.random.default_rng(core.keyed_seed(config.world_seed, TAG_CALIBRATION))
    shape = _veracity_logit_without_intercept(config.geometry, rng.random((100_000, 4)))
    low, high = -20.0, 20.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if core.sigmoid(shape + middle).mean() < config.base_rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass
class World:
    config: WorldConfig
    adjacency: sp.csr_matrix
    degree: np.ndarray
    thresholds: np.ndarray
    costs: np.ndarray
    intercept: float

    @property
    def share_cutoff(self) -> np.ndarray:
        return self.costs / (1.0 + self.costs)


def build_world(config: WorldConfig) -> World:
    if config.geometry not in GEOMETRIES or config.regime not in REGIMES:
        raise ValueError("unknown geometry or regime")
    if config.shift_credibility_weight is not None and config.geometry not in ("linear", "smooth"):
        raise ValueError("a credibility shift needs a linear or smooth geometry")
    graph = nx.watts_strogatz_graph(
        config.agents, config.degree, config.rewire,
        seed=core.keyed_seed(config.world_seed, TAG_GRAPH),
    )
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(config.agents), format="csr")
    adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
    rng = np.random.default_rng(core.keyed_seed(config.world_seed, TAG_AGENTS))
    thresholds = rng.uniform(config.threshold_low, config.threshold_high, config.agents)
    if config.regime == "A":
        costs = np.ones(config.agents)
    else:
        costs = rng.choice(np.asarray(config.cost_levels, dtype=np.float64), config.agents)
    return World(
        config, adjacency, np.asarray(adjacency.sum(axis=1)).ravel(),
        thresholds, costs, _calibrated_intercept(config),
    )


@dataclass
class Stream:
    features: np.ndarray  # [items, 4]
    veracity: np.ndarray  # [items] bool
    revealed: np.ndarray  # [items] bool
    seeds: np.ndarray  # [items, seeds_per_item]
    born: np.ndarray  # [items]
    shift_step: int | None = None

    def early_shock(self) -> float:
        """Share of false items among early items that will be fact-checked.

        The stream alone fixes it, so it is identical across arms.
        """

        early = (self.born < EARLY_SHOCK_STEPS) & self.revealed
        return float(np.mean(~self.veracity[early])) if early.any() else float("nan")


def build_stream(world: World, replicate: int) -> Stream:
    config = world.config
    count = config.steps * config.items_per_step
    rng = np.random.default_rng(core.keyed_seed(config.world_seed, TAG_ITEMS, replicate))
    features = rng.random((count, 4))
    shape = _veracity_logit_without_intercept(config.geometry, features)
    shift_step = None
    if config.shift_credibility_weight is not None:
        shift_rng = np.random.default_rng(core.keyed_seed(config.world_seed, TAG_SHIFT, replicate))
        shift_step = int(shift_rng.integers(config.steps // 4, 3 * config.steps // 4 + 1))
        after = np.repeat(np.arange(config.steps), config.items_per_step) >= shift_step
        shifted = _veracity_logit_without_intercept(
            config.geometry, features, config.shift_credibility_weight
        )
        shape = np.where(after, shifted, shape)
    truth_probability = core.sigmoid(world.intercept + shape)
    veracity = rng.random(count) < truth_probability
    revealed = rng.random(count) < config.visibility
    seeds = np.stack([
        rng.choice(config.agents, config.seeds_per_item, replace=False) for _ in range(count)
    ])
    born = np.repeat(np.arange(config.steps), config.items_per_step)
    return Stream(features, veracity, revealed, seeds, born, shift_step)


# -- policies ----------------------------------------------------------------


def hand_prior(config: WorldConfig) -> core.LinearPrior:
    """F1: trust credible items and social proof; the rest is ignored."""

    weight = np.zeros(len(config.feature_names()))
    weight[0] = 4.0  # credibility
    weight[4] = 3.0  # exposure share
    return core.LinearPrior(weight, -4.0 * 0.5 - 3.0 * 0.25)


@dataclass
class Policy:
    """How an arm turns exposures into share decisions.

    `probability(agent_ids, features, step)` returns the probability that each
    exposed item is true. `threshold_rule` replaces it with Granovetter's rule,
    which looks only at the exposure share. `observe(record, step)` lets a
    learning arm refit; it returns True when the policy changed.
    """

    name: str
    probability: Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None = None
    threshold_rule: bool = False
    observe: Callable[[core.OutcomeRecord, int], None] | None = None
    snapshots: core.PolicySnapshots = field(default_factory=core.PolicySnapshots)


def threshold_policy() -> Policy:
    return Policy("F0_threshold", threshold_rule=True)


def prior_policy(name: str, prior: core.LinearPrior) -> Policy:
    policy = Policy(name, probability=lambda _ids, x, _t: prior.probability(x))
    policy.snapshots.record(0, ("prior", prior))
    return policy


def pooled_learning_policy(
    family: str, prior: core.LinearPrior, *, seed_components: tuple[int, ...],
    learning: bool = True,
) -> Policy:
    state: dict[str, Any] = {"predictor": core.PriorPredictor(prior), "refits": 0}
    policy = Policy(f"L0_pooled_{family}")
    policy.snapshots.record(0, ("pooled", state["predictor"]))

    def probability(_ids: np.ndarray, x: np.ndarray, _t: int) -> np.ndarray:
        return state["predictor"].predict(x)

    def observe(record: core.OutcomeRecord, step: int) -> None:
        if not learning:
            return
        features, labels, _owners = record.visible(step)
        if len(labels) == 0:
            return
        seed = core.keyed_seed(*seed_components, state["refits"])
        state["refits"] += 1
        state["predictor"] = core.fit_residual(
            family, prior, features, labels, LEARNER_SETTINGS[family], seed
        )
        policy.snapshots.record(step, ("pooled", state["predictor"]))

    policy.probability = probability
    policy.observe = observe
    return policy


def per_agent_learning_policy(
    family: str, prior: core.LinearPrior, agents: int, *, seed_components: tuple[int, ...],
    device: str = "cpu", learning: bool = True,
) -> Policy:
    predictor = core.BatchedPredictor(
        prior, core.initial_batched_state(family, agents, len(prior.weight),
                                          LEARNER_SETTINGS[family].get("hidden_width", 0),
                                          core.keyed_seed(*seed_components, 0), device)
    )
    state: dict[str, Any] = {"predictor": predictor, "refits": 0}
    policy = Policy(f"L0_per_agent_{family}")
    policy.snapshots.record(0, ("per_agent", predictor))

    def probability(ids: np.ndarray, x: np.ndarray, _t: int) -> np.ndarray:
        return state["predictor"].predict_for(ids, x)

    def observe(record: core.OutcomeRecord, step: int) -> None:
        if not learning:
            return
        features, labels, owners = record.visible(step)
        if len(labels) == 0:
            return
        seed = core.keyed_seed(*seed_components, state["refits"])
        state["refits"] += 1
        fitted = core.fit_batched_residual(
            family, prior, owners, features, labels, agents=agents,
            settings=LEARNER_SETTINGS[family], seed=seed, device=device,
        )
        state["predictor"] = core.BatchedPredictor(prior, fitted)
        policy.snapshots.record(step, ("per_agent", state["predictor"]))

    policy.probability = probability
    policy.observe = observe
    return policy


def _snapshot_probability(snapshot: tuple[str, Any], ids: np.ndarray, x: np.ndarray) -> np.ndarray:
    kind, value = snapshot
    if kind == "prior":
        return value.probability(x)
    if kind == "pooled":
        return value.predict(x)
    if kind == "per_agent":
        return value.predict_for(ids, x)
    raise ValueError(f"unknown snapshot kind: {kind}")


def table_policy(name: str, donors: Sequence[core.PolicySnapshots], *, final_only: bool) -> Policy:
    """T0: replay donor policies without this run's outcomes.

    `final_only` uses each donor's last policy from the first step; otherwise
    the donor policy in force at each step. Donor probabilities are averaged.
    """

    if not donors:
        raise ValueError("a table needs at least one donor")

    def probability(ids: np.ndarray, x: np.ndarray, step: int) -> np.ndarray:
        outputs = [
            _snapshot_probability(
                donor.policies[-1] if final_only else donor.active(step), ids, x
            )
            for donor in donors
        ]
        return np.mean(outputs, axis=0)

    return Policy(name, probability=probability)


def archetype_stub_policy(
    name: str,
    archetype_of: np.ndarray,
    item_class: Callable[[np.ndarray], np.ndarray],
    answer: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    query_log: list[int],
) -> Policy:
    """X0 machinery with the LLM replaced by `answer`.

    Each step, one query is made per distinct (archetype, item class) pair with
    a pending decision, and every exposure in the pair gets that answer. The
    log records the number of queries per call, which bounds the LLM budget.
    """

    def probability(ids: np.ndarray, x: np.ndarray, _t: int) -> np.ndarray:
        keys = np.stack([archetype_of[ids], item_class(x)], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        query_log.append(len(unique))
        first = np.zeros(len(unique), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(ids))[::-1]
        return answer(unique[:, 0], unique[:, 1], ids[first], x[first])[inverse]

    return Policy(name, probability=probability)


# -- simulation ----------------------------------------------------------------


@dataclass
class RunResult:
    arm: str
    digest: str
    decisions: int
    agent_steps: int
    seconds: float
    revealed_per_agent: np.ndarray
    pooled_revealed: int
    snapshots: core.PolicySnapshots
    reach: np.ndarray  # sharers per item at the end, seeds included
    behavior: dict[str, float]


def simulate(world: World, stream: Stream, policy: Policy, replicate: int) -> RunResult:
    """Run one arm on one stream. Arms share the stream and keyed draws."""

    config = world.config
    width = len(config.feature_names())
    count = len(stream.veracity)
    shared = np.zeros((count, config.agents), dtype=bool)
    frontier = np.zeros((count, config.agents), dtype=bool)
    record = core.OutcomeRecord(width=width)
    pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    revealed_true = np.zeros(config.agents)
    revealed_total = np.zeros(config.agents)
    floor = core.ExplorationFloor(config.epsilon)
    digest = hashlib.sha256()
    decisions = 0
    true_shares = false_shares = 0
    started = time.perf_counter()

    for step in range(config.steps):
        if policy.observe is not None and step > 0 and step % config.refit_every == 0:
            policy.observe(record, step)

        # Fact-checks that become visible this step.
        for item in np.flatnonzero((stream.born + config.delay == step) & stream.revealed):
            agents, rows = pending.pop(int(item), (np.zeros(0, dtype=np.int64), None))
            if rows is None or len(agents) == 0:
                continue
            label = float(stream.veracity[item])
            record.add(rows, np.full(len(agents), label), agents, visible_from=step)
            revealed_total[agents] += 1.0
            revealed_true[agents] += label

        # New items: their seed agents share without deciding.
        for item in np.flatnonzero(stream.born == step):
            seeds = stream.seeds[item]
            shared[item, seeds] = frontier[item, seeds] = True

        active = np.flatnonzero(frontier.any(axis=1))
        if len(active) == 0:
            continue
        sharing = shared[active].T.astype(np.float64)  # [agents, active]
        exposures = np.asarray(world.adjacency @ sharing)
        new_exposure = np.asarray(world.adjacency @ frontier[active].T.astype(np.float64)) > 0
        candidate = new_exposure & ~shared[active].T
        agent_idx, column = np.nonzero(candidate)
        frontier[active] = False
        if len(agent_idx) == 0:
            continue
        items = active[column]
        exposure_count = exposures[agent_idx, column]
        columns = [
            stream.features[items],
            (exposure_count / np.maximum(world.degree[agent_idx], 1.0))[:, None],
            np.log1p(exposure_count)[:, None],
        ]
        if config.regime == "B":
            trust = (1.0 + revealed_true) / (2.0 + revealed_total)
            trust_sum = np.asarray(world.adjacency @ (sharing * trust[:, None]))
            columns.append((trust_sum[agent_idx, column] / np.maximum(exposure_count, 1.0))[:, None])
        features = np.hstack(columns)

        if policy.threshold_rule:
            share = features[:, 4] >= world.thresholds[agent_idx]
        else:
            probability = policy.probability(agent_idx, features, step)
            share = probability >= world.share_cutoff[agent_idx]
        if floor.epsilon > 0:
            for item in np.unique(items):
                rows = items == item
                draws = core.keyed_uniform(
                    (config.world_seed, TAG_EXPLORE, replicate, step, int(item)), config.agents
                )
                share[rows] |= floor.explores(draws[agent_idx[rows]])

        shared[items[share], agent_idx[share]] = True
        frontier[items[share], agent_idx[share]] = True
        decisions += len(agent_idx)
        order = np.lexsort((agent_idx, items))
        digest.update(np.stack([items[order], agent_idx[order], share[order]]).astype(np.int64).tobytes())

        for item in np.unique(items[share]):
            rows = (items == item) & share
            if stream.revealed[item] and stream.born[item] + config.delay > step:
                old_agents, old_rows = pending.get(int(item), (np.zeros(0, dtype=np.int64), np.zeros((0, width))))
                pending[int(item)] = (
                    np.concatenate([old_agents, agent_idx[rows]]),
                    np.vstack([old_rows, features[rows]]),
                )
            truth = bool(stream.veracity[item])
            true_shares += int(rows.sum()) if truth else 0
            false_shares += 0 if truth else int(rows.sum())

    seconds = time.perf_counter() - started
    final_counts = record.visible_counts_by_agent(config.steps, config.agents)
    total = true_shares + false_shares
    return RunResult(
        arm=policy.name,
        digest=digest.hexdigest(),
        decisions=decisions,
        agent_steps=config.agents * config.steps,
        seconds=seconds,
        revealed_per_agent=final_counts,
        pooled_revealed=int(final_counts.sum()),
        snapshots=policy.snapshots,
        reach=shared.sum(axis=1),
        behavior={  # never reported at stage D1
            "share_precision": true_shares / total if total else float("nan"),
            "false_shares": float(false_shares),
            "false_reach": float(shared.sum(axis=1)[~stream.veracity].mean())
            if (~stream.veracity).any() else float("nan"),
        },
    )


# -- archetypes ----------------------------------------------------------------


def archetypes(world: World, threshold_bins: int, cost_bins: bool) -> np.ndarray:
    """Group agents by threshold quantile and, in Regime B, by cost level.

    Degree is nearly constant on a Watts-Strogatz graph, so it cannot group
    agents; the threshold and the cost are the attributes that differ.
    """

    edges = np.quantile(world.thresholds, np.linspace(0, 1, threshold_bins + 1)[1:-1])
    group = np.searchsorted(edges, world.thresholds, side="right")
    if cost_bins:
        levels = {value: index for index, value in enumerate(sorted(set(world.costs)))}
        group = group * len(levels) + np.asarray([levels[c] for c in world.costs])
    return group.astype(np.int64)


def item_classes(bins_per_feature: int, features_used: int = 2) -> Callable[[np.ndarray], np.ndarray]:
    """Bin the first `features_used` item features into equal-width classes."""

    def classify(x: np.ndarray) -> np.ndarray:
        binned = np.minimum((x[:, :features_used] * bins_per_feature).astype(np.int64), bins_per_feature - 1)
        return (binned * (bins_per_feature ** np.arange(features_used))).sum(axis=1)

    return classify


# -- feasibility profile ---------------------------------------------------------


PROFILES = {
    "smoke": {"agents": 200, "steps": 24, "items_per_step": 8, "replicates": 1, "donors": 2},
    "feasibility": {"agents": 1000, "steps": 60, "items_per_step": 20, "replicates": 2, "donors": 4},
}


def _task(args: tuple[str, str, str, int]) -> dict[str, Any]:
    profile, regime, geometry, world_seed = args
    sizes = PROFILES[profile]
    config = WorldConfig(
        agents=sizes["agents"], steps=sizes["steps"], items_per_step=sizes["items_per_step"],
        regime=regime, geometry=geometry, world_seed=world_seed, epsilon=0.02,
    )
    world = build_world(config)
    prior = hand_prior(config)
    families = ("linear", "mlp", "tree") if regime == "A" else ("linear", "mlp")
    rows: list[dict[str, Any]] = []

    def learner(family: str, replicate: int) -> Policy:
        components = (world_seed, ord(regime), GEOMETRIES.index(geometry), replicate, FAMILIES_INDEX[family])
        if regime == "A":
            return pooled_learning_policy(family, prior, seed_components=components)
        return per_agent_learning_policy(family, prior, config.agents, seed_components=components)

    donor_replicates = range(1000, 1000 + sizes["donors"])
    for family in families:
        donors = [
            simulate(world, build_stream(world, r), learner(family, r), r).snapshots
            for r in donor_replicates
        ]
        for replicate in range(sizes["replicates"]):
            stream = build_stream(world, replicate)
            arms = [
                learner(family, replicate),
                table_policy(f"T0_final_{family}", donors, final_only=True),
                table_policy(f"T0_traj_{family}", donors, final_only=False),
            ]
            for policy in arms:
                result = simulate(world, stream, policy, replicate)
                rows.append(_feasibility_row(config, result, replicate))
    for replicate in range(sizes["replicates"]):
        stream = build_stream(world, replicate)
        for policy in (threshold_policy(), prior_policy("F1_hand", prior)):
            rows.append(_feasibility_row(config, simulate(world, stream, policy, replicate), replicate))
        for threshold_bins in (2, 4):
            for bins in (2, 4):
                log: list[int] = []
                stub = archetype_stub_policy(
                    f"X0_stub_k{threshold_bins}_b{bins}",
                    archetypes(world, threshold_bins, cost_bins=regime == "B"),
                    item_classes(bins),
                    lambda _a, _c, _ids, x: prior.probability(x),
                    log,
                )
                result = simulate(world, stream, stub, replicate)
                row = _feasibility_row(config, result, replicate)
                row["archetype_queries"] = int(sum(log))
                rows.append(row)
    return {"regime": regime, "geometry": geometry, "world_seed": world_seed, "rows": rows}


FAMILIES_INDEX = {"linear": 1, "mlp": 2, "tree": 3}


def _feasibility_row(config: WorldConfig, result: RunResult, replicate: int) -> dict[str, Any]:
    per_agent = result.revealed_per_agent
    return {
        "arm": result.arm,
        "replicate": replicate,
        "decisions": result.decisions,
        "seconds": round(result.seconds, 4),
        "microseconds_per_agent_step": round(1e6 * result.seconds / result.agent_steps, 3),
        "pooled_revealed_outcomes": result.pooled_revealed,
        "revealed_per_agent": {
            "median": float(np.median(per_agent)),
            "q10": float(np.quantile(per_agent, 0.1)),
            "share_with_at_least_5": float(np.mean(per_agent >= 5)),
            "share_with_at_least_20": float(np.mean(per_agent >= 20)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    tasks = [(args.profile, regime, geometry, 0) for regime in REGIMES for geometry in ("linear", "axis", "null")]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.workers <= 1:
        results = [_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_task, tasks))
    summary = {
        "status": "D1_FEASIBILITY_ONLY_NO_BEHAVIORAL_OUTCOME_REPORTED",
        "profile": args.profile,
        "profile_sizes": PROFILES[args.profile],
        "wall_seconds": round(time.perf_counter() - started, 2),
        "tasks": results,
    }
    (args.output_dir / "feasibility.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}))


if __name__ == "__main__":
    main()
