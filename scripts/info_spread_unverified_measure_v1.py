"""The unverified-item measure for the information-spread policy ladder.

The user chose on 2026-10-18 to make the primary measure share precision on
items whose fact-check is not public at the decision. D4 had shown that an
MLP learner recognizes items it has seen fact-checked, so precision over all
items rewards that recognition.

The frozen simulator is not changed. A wrapper logs every decision of a
probability policy, reapplies the exploration floor's keyed draws exactly as
the simulator does, and classifies each decision by the item's fact-check
state at that step:

- unrevealed: the item is never fact-checked;
- pending: it will be, but the check is not yet public;
- visible: its check is already public.

Unverified means unrevealed or pending. Every run checks that the
reconstructed shares per item and the decision count equal the simulator's
own, so the measure counts exactly the decisions the simulator made.
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

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


STATES = ("unrevealed", "pending", "visible")


def threshold_probability_policy(world: spread.World) -> spread.Policy:
    """F0 in probability form: 1 when the exposure share reaches the agent's
    threshold, else 0.

    Every cutoff c/(1 + c) lies strictly between 0 and 1, so this makes the
    decisions of `spread.threshold_policy` exactly.
    """

    def probability(ids: np.ndarray, x: np.ndarray, _t: int) -> np.ndarray:
        return (x[:, 4] >= world.thresholds[ids]).astype(np.float64)

    return spread.Policy("F0_threshold", probability=probability)


def _item_index(stream: spread.Stream) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(stream.features[:, 0], kind="stable")
    return order, stream.features[order, 0]


def measured_simulate(world: spread.World, stream: spread.Stream, policy: spread.Policy,
                      replicate: int) -> tuple[spread.RunResult, dict[str, Any]]:
    """Run the frozen simulator and measure precision by fact-check state."""

    if policy.probability is None:
        raise ValueError("a policy without probabilities: use threshold_probability_policy for F0")
    config = world.config
    floor = core.ExplorationFloor(config.epsilon)
    order, sorted_credibility = _item_index(stream)
    count = len(stream.veracity)
    shares = np.zeros(len(STATES), dtype=np.int64)
    true = np.zeros(len(STATES), dtype=np.int64)
    decisions = np.zeros(len(STATES), dtype=np.int64)
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
        true[:] += np.bincount(state[share & stream.veracity[items]], minlength=len(STATES))
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
    }
    return result, measures
