"""The agents' utility on unverified items, for the information-spread ladder (O25).

Share precision rewards selectivity: a policy that shares fewer, surer items
scores higher even when it forgoes shares worth making. The agents' own
objective is different. A share of a true item gains 1, a share of a false
item costs the sharer c, and a decline is worth 0. The decision rule (share
when the probability of truth reaches c/(1 + c)) maximizes this utility for a
calibrated belief.

This module measures that utility on unverified items, whose fact-check is
not public at the decision, as the unverified measure does for precision. It
keeps the unverified measure's reconstruction (scripts/
info_spread_unverified_measure_v1.py, which is frozen):

- a wrapper logs each decision;
- it reapplies the exploration floor's keyed draws;
- every run checks the reconstructed shares per item, and the decision count,
  against the simulator's own.

It adds the cost of every false share, by sharer.

Utility per agent is (true shares - summed costs of false shares) / agents,
on unverified items. In Regime A every cost is 1, so it is
(true - false) / agents, computable from the recorded counts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import info_spread_unverified_measure_v1 as unverified  # noqa: E402
import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


STATES = unverified.STATES


def utility_from_counts(by_state: dict[str, Any], agents: int) -> float:
    """Unit-cost utility per agent on unverified items, from recorded counts."""

    shares = by_state["unrevealed"]["shares"] + by_state["pending"]["shares"]
    true = by_state["unrevealed"]["true"] + by_state["pending"]["true"]
    return (true - (shares - true)) / agents


def measured_simulate(world: spread.World, stream: spread.Stream, policy: spread.Policy,
                      replicate: int) -> tuple[spread.RunResult, dict[str, Any]]:
    """The unverified measure, plus the cost of false shares by state."""

    if policy.probability is None:
        raise ValueError("a policy without probabilities: use threshold_probability_policy for F0")
    config = world.config
    floor = core.ExplorationFloor(config.epsilon)
    order = np.argsort(stream.features[:, 0], kind="stable")
    sorted_credibility = stream.features[order, 0]
    count = len(stream.veracity)
    shares = np.zeros(len(STATES), dtype=np.int64)
    true = np.zeros(len(STATES), dtype=np.int64)
    decisions = np.zeros(len(STATES), dtype=np.int64)
    false_cost = np.zeros(len(STATES), dtype=np.float64)
    per_item = np.zeros(count, dtype=np.int64)
    inner = policy.probability

    def probability(ids: np.ndarray, x: np.ndarray, step: int) -> np.ndarray:
        p = inner(ids, x, step)
        items = order[np.searchsorted(sorted_credibility, x[:, 0])]
        if not np.array_equal(stream.features[items], x[:, :4]):
            raise RuntimeError("an exposure does not match a stream item")
        share = p >= world.share_cutoff[ids]
        if floor.epsilon > 0:
            for item in np.unique(items):
                rows = items == item
                draws = core.keyed_uniform(
                    (config.world_seed, spread.TAG_EXPLORE, replicate, step, int(item)), config.agents)
                share[rows] |= floor.explores(draws[ids[rows]])
        state = np.where(~stream.revealed[items], 0, np.where(stream.born[items] + config.delay > step, 1, 2))
        decisions[:] += np.bincount(state, minlength=len(STATES))
        shares[:] += np.bincount(state[share], minlength=len(STATES))
        is_true = stream.veracity[items]
        true[:] += np.bincount(state[share & is_true], minlength=len(STATES))
        false_shares = share & ~is_true
        false_cost[:] += np.bincount(state[false_shares], weights=world.costs[ids[false_shares]],
                                     minlength=len(STATES))
        per_item[:] += np.bincount(items[share], minlength=count)
        return p

    policy.probability = probability
    try:
        result = spread.simulate(world, stream, policy, replicate)
    finally:
        policy.probability = inner
    if int(decisions.sum()) != result.decisions or not np.array_equal(per_item, result.decision_shares):
        raise RuntimeError("the reconstructed decisions differ from the simulator's")
    unverified_shares = int(shares[0] + shares[1])
    unverified_true = int(true[0] + true[1])
    measures = {
        "share_precision": result.behavior["share_precision"],
        "unverified_share_precision": unverified_true / unverified_shares if unverified_shares else float("nan"),
        "false_reach": result.behavior["false_reach"],
        "decisions": result.decisions,
        "by_state": {name: {"decisions": int(decisions[i]), "shares": int(shares[i]), "true": int(true[i])}
                     for i, name in enumerate(STATES)},
        "unverified_utility_per_agent": (unverified_true - float(false_cost[0] + false_cost[1])) / config.agents,
    }
    return result, measures
