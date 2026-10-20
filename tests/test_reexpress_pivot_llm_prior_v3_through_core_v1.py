from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import reexpress_pivot_llm_prior_v3_through_core_v1 as rx  # noqa: E402


# A stand-in for the hash-bound LLM prior, which lives outside the repository.
SYNTHETIC_LLM_PRIOR = {"weights": [0.8, -0.3, 0.5, -0.6, -0.4, 0.2, 0.3, 0.1, 0.4], "bias": -1.2}


def test_the_core_draw_is_pivots_exploration_draw() -> None:
    for components in ((20261023, 0, 83, 3, 5), (20261025, 7, 83, 17, 255)):
        pivot = float(np.random.default_rng(np.random.SeedSequence(components)).random())
        assert core.keyed_uniform(components, 1)[0] == pivot
    assert core.keyed_seed(1, 2, 3, 0) == int(np.random.SeedSequence((1, 2, 3, 0)).generate_state(1)[0])


def test_the_hand_rule_as_a_core_prior_gives_pivots_starting_output() -> None:
    contract = rx.pilot3.load_contract()
    condition = rx.pilot3._condition(contract, "hand")
    features = np.random.default_rng(3).random((64, len(rx.pilot1.FEATURES)))
    rule = rx.core_prior(condition, SYNTHETIC_LLM_PRIOR)
    assert np.array_equal(rule.probability(features), rx.abm1._prior_output(features))


@pytest.mark.parametrize("condition_id", ["hand", "hybrid_20", "flat"])
def test_a_smoke_task_through_the_core_makes_pivots_rows(condition_id: str) -> None:
    contract = rx.pilot3.load_contract()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        pivot = rx.pilot3.run_task(contract, "smoke", 0, "smooth_interaction_fit", condition_id, SYNTHETIC_LLM_PRIOR)
        through_core = rx.run_task_core(contract, "smoke", 0, "smooth_interaction_fit", condition_id,
                                        SYNTHETIC_LLM_PRIOR)
    assert [row["arm_id"] for row in through_core] == ["fixed_rule", "linear_batch", "tree_batch", "mlp_batch"]
    assert [json.dumps(row, sort_keys=True) for row in through_core] == [
        json.dumps(row, sort_keys=True) for row in pivot]
    assert rx.abm2.ProbedLearner.__name__ == "ProbedLearner"


def test_compare_reports_each_mismatched_field() -> None:
    recorded = [{"replicate": 0, "world_id": "w", "condition_id": "c", "arm_id": "a", "x": 1.0, "y": 2.0}]
    rows = [dict(recorded[0], y=2.5)]
    report = rx.compare(rows, recorded)
    assert report["identical_rows"] == 0 and report["mismatched"] == {"0/w/c/a": ["y"]}
    assert rx.compare(recorded, recorded)["identical_rows"] == 1
