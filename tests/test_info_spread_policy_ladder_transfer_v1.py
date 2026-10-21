from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_transfer_v1 as transfer  # noqa: E402


def _contracts() -> tuple[dict, dict]:
    contract = transfer.load_contract()
    return contract, transfer.load_v2_contract(contract)


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract, _v2 = _contracts()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(transfer.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-transfer-v1.md",
        "experiments/configs/info_spread_policy_ladder_transfer_v1.yaml",
        "scripts/run_info_spread_policy_ladder_transfer_v1.py",
        "tests/test_info_spread_policy_ladder_transfer_v1.py",
    }
    with pytest.raises(RuntimeError):
        transfer.verify_precommit("abc123", contract)


def test_donor_worlds_are_other_worlds_with_disjoint_seeds() -> None:
    contract, v2_contract = _contracts()
    cells = int(v2_contract["profiles"]["development"]["cells"])
    count = int(contract["donors"]["count"])
    seeds = {transfer.donor_world_seed(contract, c, d) for c in range(cells) for d in range(count)}
    assert len(seeds) == cells * count
    study_seeds = {int(v2_contract["seeds"]["world_seed_offset"]) + c for c in range(cells)}
    assert not seeds & study_seeds
    assert min(seeds) >= 9000 and not seeds & set(range(5000, 8000))
    learner_cells = {transfer.donor_cell_id(contract, c, d) for c in range(cells) for d in range(count)}
    assert len(learner_cells) == cells * count and min(learner_cells) >= cells
    donor = transfer.build_donor_world(contract, v2_contract, "smoke", "B", "linear", 0, 1)
    target = v2.build_world(v2_contract, "smoke", "B", "linear", 0)
    assert donor.config.geometry == target.config.geometry and donor.config.regime == target.config.regime
    assert donor.config.world_seed != target.config.world_seed
    assert (donor.costs != target.costs).any() or (donor.thresholds != target.thresholds).any()


def _table(positive: dict[str, int], null_material: bool = False) -> dict:
    _contract, v2_contract = _contracts()
    design = v2_contract["design"]
    table = {}
    for regime in design["regimes"]:
        study = [f"{regime}/{f}/{g}" for f in design["families"][regime] for g in design["study_geometries"]]
        for index, key in enumerate(study):
            win = index < positive[regime]
            rule = {"material": win, "mean_cell_delta": 0.05 if win else 0.0}
            table[key] = {name: rule for name in transfer.COMPARISONS}
        for family in design["families"][regime]:
            rule = {"material": null_material, "mean_cell_delta": -0.05 if null_material else 0.0}
            table[f"{regime}/{family}/null"] = {name: rule for name in transfer.COMPARISONS}
    return table


@pytest.mark.parametrize(
    ("positive", "expected"),
    [({"A": 0, "B": 5}, {"A": "TRANSFERRED_TABLE_SUFFICES", "B": "ADAPTIVE_BEATS_TRANSFERRED_TABLE"}),
     ({"A": 1, "B": 4}, {"A": "TRANSFERRED_TABLE_SUFFICES", "B": "ADAPTIVE_BEATS_TRANSFERRED_TABLE"}),
     ({"A": 2, "B": 3}, {"A": "MIXED", "B": "MIXED"})],
)
def test_the_registered_dispositions(positive: dict, expected: dict) -> None:
    contract, v2_contract = _contracts()
    result = transfer.dispositions(contract, v2_contract, _table(positive))
    assert result["transfer_best_table"] == expected
    assert result["design_check"] == "NULL_WORLD_PASSES"
    defect = transfer.dispositions(contract, v2_contract, _table(positive, null_material=True))
    assert defect["design_check"] == "DESIGN_DEFECT_IN_NULL_WORLD"  # either direction counts


def test_a_smoke_study_runs_deterministically_and_passes_parity() -> None:
    contract, v2_contract = _contracts()
    v2.apply_learner_settings(v2_contract)
    v2_cells, cells = {}, []
    for regime in v2_contract["design"]["regimes"]:
        for family in v2_contract["design"]["families"][regime]:
            first = transfer.run_cell(contract, v2_contract, "smoke", regime, family, "linear", 0, "cpu")
            second = transfer.run_cell(contract, v2_contract, "smoke", regime, family, "linear", 0, "cpu")
            assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
            assert set(first["rows"][0]["arms"]) == set(transfer.ARMS)
            assert "seconds" not in json.dumps(first)
            cells.append(first)
            v2_cells[(regime, family, "linear", 0)] = v2.run_cell(v2_contract, "smoke", regime, family, "linear", 0, "cpu")
    summary = transfer.summarize(contract, v2_contract, v2_cells, cells)
    assert summary["parity_failures"] == []
    entry = summary["table"]["B/partial_mlp/linear"]
    assert set(entry[v2.PRIMARY]) == {*transfer.COMPARISONS, *transfer.DESCRIPTIVE}
    assert set(entry[f"mean_{v2.PRIMARY}"]) == {*v2.ARMS, *transfer.ARMS}
