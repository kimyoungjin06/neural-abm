from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_pivot_empirical_ladder_power_levers_v1 as screen  # noqa: E402


def test_the_screen_binds_v3_and_uses_an_unregistered_seed() -> None:
    contract = screen.load_contract()
    seeds = contract["seeds"]
    used = {int(seeds["base_seed"]), int(seeds["standardization_seed"]), *(int(s) for s in seeds["opened"])}
    assert screen.EXPLORATION_SEED not in used


def test_lever_c_lifts_the_g4_guard_by_the_recorded_removal_counts() -> None:
    contract = screen.load_contract()
    base, _ = screen.shapes_for(contract, screen.CONFIGS["base"], "development")
    lifted, _ = screen.shapes_for(contract, screen.CONFIGS["C_no_g4_guard"], "development")
    assert screen.CONFIGS["C_no_g4_guard"]["entrants_scale"] == pytest.approx(1.7071, abs=1e-4)
    assert sum(s["entrants"] for s in lifted) / sum(s["entrants"] for s in base) == pytest.approx(1.7071, abs=1e-3)
    assert sum(s["at_risk"] for s in lifted) / sum(s["at_risk"] for s in base) == pytest.approx(1.2746, abs=1e-3)


def test_the_clustered_test_recovers_a_known_last_coefficient() -> None:
    rng = np.random.default_rng(0)
    n, groups = 40_000, 40
    x = rng.standard_normal((n, 2))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 0.2 * x[:, 0] + 0.5 * x[:, 1])))).astype(float)
    rows = {"y": y, "w": np.ones(n), "episode": np.repeat(np.arange(groups), n // groups)}
    result = screen.clustered_test(x, rows)
    assert result["coefficient"] == pytest.approx(0.5, abs=0.05)
    assert result["won"] is True and 0 < result["se"] < 0.05
    null = screen.clustered_test(np.column_stack([x[:, 1], rng.standard_normal(n)]), rows)
    assert abs(null["coefficient"]) < 3 * null["se"]


def test_the_joint_design_puts_the_learned_term_last() -> None:
    n = 5
    rows = {"attention": np.zeros(n), "x": np.zeros((n, 4)), "quarter": np.linspace(0, 1, n),
            screen.ANTICIPATORY: np.arange(n, dtype=float), screen.LEARNED: np.full(n, 7.0)}
    assert screen.joint_design(rows, False).shape == (n, 7)
    ramp = screen.joint_design(rows, True)
    assert ramp.shape == (n, 8) and np.all(ramp[:, -1] == 7.0)
    assert np.allclose(ramp[:, -2], rows[screen.ANTICIPATORY] * rows["quarter"])
