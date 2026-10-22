from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_flip_rate_v1 as flip  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402


def _contracts() -> tuple[dict, dict]:
    contract = flip.load_contract()
    return contract, flip.load_v2_contract(contract)


def _streams_equal(a: spread.Stream, b: spread.Stream) -> bool:
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("features", "veracity", "revealed", "seeds", "born"))


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract, _v2 = _contracts()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(flip.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-flip-rate-v1.md",
        "experiments/configs/info_spread_policy_ladder_flip_rate_v1.yaml",
        "scripts/run_info_spread_policy_ladder_flip_rate_v1.py",
        "tests/test_info_spread_policy_ladder_flip_rate_v1.py",
    }
    assert contract["seeds"]["world_seed_offset"] == 11000


def test_the_ends_reproduce_the_prototypes_streams() -> None:
    contract, v2_contract = _contracts()
    world = flip.build_world(contract, v2_contract, "smoke", "A", 3)
    signed = spread.build_world(dataclasses.replace(world.config, random_credibility_sign=True))
    for replicate in range(12):
        stationary = flip.build_stream(world, replicate, 0.0)
        assert _streams_equal(stationary, spread.build_stream(world, replicate))
        assert stationary.credibility_sign == 1
        coin = flip.build_stream(world, replicate, 0.5)
        frozen = spread.build_stream(signed, replicate)
        assert _streams_equal(coin, frozen) and coin.credibility_sign == frozen.credibility_sign
    stream = spread.build_stream(world, 2)
    policy = spread.prior_policy("F1", spread.hand_prior(world.config))
    assert (spread.simulate(world, flip.build_stream(world, 2, 0.0), policy, 2).digest
            == spread.simulate(world, stream, spread.prior_policy("F1", spread.hand_prior(world.config)), 2).digest)


def test_flips_occur_at_the_rate_and_nest_across_rates() -> None:
    contract, v2_contract = _contracts()
    world = flip.build_world(contract, v2_contract, "smoke", "B", 0)
    count = 2000
    signs = {rate: np.array([flip.build_stream(world, r, rate).credibility_sign for r in range(count)])
             for rate in (0.1, 0.25, 0.5)}
    for rate, values in signs.items():
        assert abs(float(np.mean(values == -1)) - rate) < 4 * np.sqrt(rate * (1 - rate) / count)
    assert np.all((signs[0.1] == -1) <= (signs[0.25] == -1))
    assert np.all((signs[0.25] == -1) <= (signs[0.5] == -1))
    with pytest.raises(ValueError):
        flip.build_stream(spread.build_world(dataclasses.replace(world.config, random_credibility_sign=True)), 0, 0.1)


def _table(onsets: dict[str, float | None], control_ok: bool = True) -> dict:
    contract, _v2 = _contracts()
    table = {}
    for regime, families in contract["design"]["families"].items():
        for family in families:
            onset = onsets.get(f"{regime}/{family}")
            for rate in contract["design"]["flip_rates"]:
                win = onset is not None and float(rate) >= onset
                best = {"material": win, "mean_cell_delta": 0.05 if win else 0.0}
                closed = {"material": control_ok or float(rate) < 0.5, "mean_cell_delta": 0.05}
                table[flip.group_key(regime, family, rate)] = {
                    "best_table": best, "closed_loop": closed, "hand_rule": {"material": False, "mean_cell_delta": 0.0}}
    return table


def test_the_registered_onsets_and_control() -> None:
    contract, _v2 = _contracts()
    result = flip.dispositions(contract, _table({"A/linear_exact": 0.25, "A/mlp": 0.5, "B/partial_linear_exact": 0.1}))
    assert result["onset"] == {"A/linear_exact": "0.25", "A/mlp": "0.5", "B/partial_linear_exact": "0.1",
                               "B/partial_mlp": "NONE_UP_TO_0.5"}
    assert result["design_check"] == "POSITIVE_CONTROL_HOLDS_AT_0.5"
    failed = flip.dispositions(contract, _table({}, control_ok=False))
    assert failed["design_check"] == "POSITIVE_CONTROL_FAILS_AT_0.5" and len(failed["control_failures"]) == 4


def test_a_smoke_cell_runs_deterministically_for_every_family() -> None:
    contract, v2_contract = _contracts()
    v2.apply_learner_settings(v2_contract)
    cells = []
    for regime, families in contract["design"]["families"].items():
        for family in families:
            for rate in (0.0, 0.5):
                first = flip.run_cell(contract, v2_contract, "smoke", regime, family, rate, 0, "cpu")
                second = flip.run_cell(contract, v2_contract, "smoke", regime, family, rate, 0, "cpu")
                assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
                assert "seconds" not in json.dumps(first)
                cells.append(first)
    summary = flip.summarize(contract, cells)
    assert set(summary["table"]) == {flip.group_key(r, f, p) for r, fs in contract["design"]["families"].items()
                                     for f in fs for p in (0.0, 0.5)}
    assert summary["table"]["A/mlp/p=0"]["flipped_run_share"] == 0.0
