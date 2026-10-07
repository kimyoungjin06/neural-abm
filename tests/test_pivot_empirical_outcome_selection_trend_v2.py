from __future__ import annotations

import ast
import copy
import datetime as dt
import importlib.util
import itertools
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_outcome_selection_trend_v2.py"
CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_outcome_selection_trend_v2.yaml"


def _module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("n10_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n10 = _module()


def _contract() -> dict:
    return n10.load_contract(CONFIG)


def test_v2_amends_only_the_clock_persistence_alignment_and_transfer_levels() -> None:
    contract = _contract()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert contract["amends"]["result_informed_successor"] is True
    assert "ENTRY_CLOCK_PUBLICATION_DATE_QUARTERS" in contract["amends"]["changed"]
    assert len(n10.v1.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert contract["threshold_transfer"]["empirical_entry_levels"] == 12
    assert contract["boundaries"]["empirical_rows_read"] == 0


@pytest.mark.parametrize("block", ["fields", "years", "episodes", "partition", "precision", "decision", "claims"])
def test_v2_may_not_change_a_v1_block(block: str) -> None:
    contract = copy.deepcopy(_contract())
    contract[block] = {**contract[block], "changed": True}
    with pytest.raises(ValueError, match="may not change"):
        n10.validate_contract(contract)


def test_v2_keeps_the_fine_threshold_and_needs_publication_date() -> None:
    contract = copy.deepcopy(_contract())
    contract["threshold_transfer"]["fine_clock_threshold"] = 0.05
    with pytest.raises(ValueError, match="fine_clock_threshold"):
        n10.validate_contract(contract)
    contract = copy.deepcopy(_contract())
    contract["source"]["required_columns"].remove("publication_date")
    with pytest.raises(ValueError, match="publication_date"):
        n10.validate_contract(contract)


def test_the_v1_script_is_bound_by_hash() -> None:
    assert n10._sha256(n10.V1_SCRIPT) == n10.V1_SCRIPT_SHA256
    assert n10.v1.spearman is not None


def test_the_quarter_clock_and_the_january_first_rule() -> None:
    assert n10.quarter_index(dt.date(2005, 3, 31)) == 4 * 2005
    assert n10.quarter_index(dt.date(2005, 4, 1)) == 4 * 2005 + 1
    clock = n10.entry_clock([dt.date(2006, 8, 3), dt.date(2006, 2, 9)], 2005)
    assert (clock["offset"], clock["response_year"], clock["precise"]) == (4, 1, True)
    assert n10.entry_clock([dt.date(2007, 1, 1), dt.date(2007, 5, 2)], 2005)["precise"] is False
    assert n10.entry_clock([dt.date(2007, 12, 31)], 2005)["offset"] == 11
    for date in (dt.date(2004, 12, 31), dt.date(2008, 1, 2)):
        with pytest.raises(ValueError):
            n10.entry_clock([date], 2005)


def test_persistence_is_aligned_to_the_entry_quarter() -> None:
    entry = dt.date(2005, 2, 10)
    assert n10.persisted(entry, [dt.date(2005, 3, 30)]) is False
    assert n10.persisted(entry, [dt.date(2005, 4, 1)]) is True
    assert n10.persisted(entry, [dt.date(2008, 3, 31)]) is True
    assert n10.persisted(entry, [dt.date(2008, 4, 1)]) is False
    late = dt.date(2005, 11, 20)
    assert n10.persisted(late, [dt.date(2008, 12, 31)]) is True
    assert n10.persisted(late, [dt.date(2006, 1, 1)], first=9) is False
    assert n10.persisted(late, [dt.date(2006, 1, 1), dt.date(2007, 6, 1)], minimum_works=2) is True


def test_every_entry_quarter_gets_the_same_window_length() -> None:
    for month in (1, 4, 7, 10):
        entry = dt.date(2005, month, 15)
        start = n10.quarter_index(entry)
        dates = [dt.date(year, m, 15) for year in range(2005, 2010) for m in (1, 4, 7, 10)]
        counted = [d for d in dates if n10.persisted(entry, [d])]
        assert len(counted) == 12
        assert min(n10.quarter_index(d) for d in counted) == start + 1


def test_the_episode_statistic_uses_precise_entrants_and_response_years() -> None:
    validity = _contract()["episode_validity"]
    entrants = [{"offset": k, "persisted": p, "precise": True} for k, p in itertools.product(range(12), (0, 1))] * 2
    entrants += [{"offset": 0, "persisted": 0, "precise": False}] * 6
    result = n10.episode_statistic(entrants, validity)
    assert (result["precise_entrants"], result["valid"]) == (48, True)
    assert result["imprecise_share"] == pytest.approx(6 / 54)
    one_year = [{"offset": k % 4, "persisted": k % 2, "precise": True} for k in range(40)]
    assert n10.episode_statistic(one_year, validity)["meets_floor"] is False


def test_the_module_has_no_empirical_row_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"pyarrow", "pandas", "polars", "duckdb"}
    assert "pivot-feedback-lean-h3-openalex-full-core" not in source


def test_twelve_level_bins_cover_the_synthetic_window() -> None:
    bins = [n10.v1.coarsen(step, 3, 20, 12) for step in range(3, 21)]
    assert sorted(set(bins)) == list(range(12))
    assert bins == sorted(bins)


@pytest.mark.skipif(
    not Path("/ssd/pivot_observable_one_sided_v1_c0a8b1a_development/runs.jsonl").exists(),
    reason="the synthetic runs are machine-local",
)
def test_the_stored_runs_reconstruct_the_fine_clock_statistic() -> None:
    transfer = _contract()["threshold_transfer"]
    for spec in transfer["studies"]:
        study_contract = n10.yaml.safe_load((ROOT / spec["contract"]).read_text(encoding="utf-8"))
        geometry = n10.v1.study_geometry(study_contract, transfer["profile"])
        with Path(spec["runs"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in itertools.islice(handle, 100)]
        assert all(n10.v1.run_statistics(row, geometry, 12, 1e-12)["reconstructed"] for row in rows)
