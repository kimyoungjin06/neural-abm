from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_info_spread_null_leak_v1 as leak  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as d4  # noqa: E402


def _world(regime: str) -> spread.World:
    return spread.build_world(spread.WorldConfig(
        agents=60, steps=20, items_per_step=4, regime=regime, geometry="null", epsilon=0.02, world_seed=3,
    ))


def test_logging_changes_no_decision_and_splits_every_one() -> None:
    contract = d4.load_contract()
    d4.apply_learner_settings(contract)
    world = _world("B")
    stream = spread.build_stream(world, 0)
    plain = spread.simulate(world, stream, d4._learner(contract, world, "partial_mlp", 3, 0, 0, "cpu"), 0)
    log: list[np.ndarray] = []
    policy = leak.logged(d4._learner(contract, world, "partial_mlp", 3, 0, 0, "cpu"), world, stream, log)
    wrapped = spread.simulate(world, stream, policy, 0)
    assert wrapped.digest == plain.digest
    split = leak.split_precision(world, stream, log)
    assert sum(v["decisions"] for v in split.values()) == wrapped.decisions
    assert split["visible"]["decisions"] > 0


def test_without_trust_holds_the_trust_feature_constant() -> None:
    world = _world("B")
    stream = spread.build_stream(world, 1)
    seen: list[np.ndarray] = []

    def probability(_ids: np.ndarray, x: np.ndarray, _t: int) -> np.ndarray:
        seen.append(x[:, leak.TRUST_COLUMN].copy())
        return np.full(len(x), 0.6)

    observed: list[np.ndarray] = []
    policy = spread.Policy("probe", probability=probability,
                           observe=lambda record, _step: observed.append(np.asarray(record.features)))
    spread.simulate(world, stream, leak.without_trust(policy), 1)
    assert all(np.all(values == leak.TRUST_CONSTANT) for values in seen)
    assert any(len(rows) for rows in observed)
    assert all(np.all(rows[:, leak.TRUST_COLUMN] == leak.TRUST_CONSTANT) for rows in observed if len(rows))
