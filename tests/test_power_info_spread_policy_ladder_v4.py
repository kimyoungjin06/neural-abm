from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import power_info_spread_policy_ladder_v4 as power4  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as d4  # noqa: E402


SIZES = {"agents": 60, "steps": 24, "items_per_step": 4, "epsilon": 0.02}


def test_every_control_runs_deterministically_at_a_tiny_size() -> None:
    contract = d4.load_contract()
    d4.apply_learner_settings(contract)
    cells = []
    for control in power4.CONTROLS:
        for regime in ("A", "B"):
            family = power4.FAMILIES[regime][1 if control == "NULL" else 0]
            first = power4.run_cell(contract, control, regime, family, 0, replicates=2, sizes=SIZES)
            second = power4.run_cell(contract, control, regime, family, 0, replicates=2, sizes=SIZES)
            assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
            cells.append(first)
    assert "seconds" not in json.dumps(cells)
    summary = power4.summarize(cells)
    assert set(summary["reading"]) == {
        "neg_no_material_pair", "pos_sign_closed_loop_material_and_positive",
        "null_no_material_positive_learner_gain", "adequate", "null_old_measure_material_positive_groups",
    }
    entry = summary["table"]["NULL/B/partial_mlp/L0-F1"]
    assert set(entry) == {"cells", *power4.MEASURES}


def test_null_worlds_are_null_and_sign_worlds_draw_a_sign() -> None:
    contract = d4.load_contract()
    null = power4.build_world(contract, "NULL", "A", 0, SIZES)
    sign = power4.build_world(contract, "POS_sign", "A", 0, SIZES)
    assert null.config.geometry == "null" and not null.config.random_credibility_sign
    assert sign.config.geometry == "linear" and sign.config.random_credibility_sign
    assert null.config.world_seed == power4.CELL_SEED_OFFSET
