from __future__ import annotations

import ast
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_outcome_selection_trend_v1.py"
CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_outcome_selection_trend_v1.yaml"


def _module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("n10", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n10 = _module()


def _contract() -> dict:
    return n10.load_contract(CONFIG)


def test_the_contract_is_a_precommit_and_every_binding_holds() -> None:
    contract = _contract()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(n10.verify_source_bindings(contract)) == len(contract["source_bindings"])
    boundaries = contract["boundaries"]
    assert boundaries["empirical_rows_read"] == 0
    assert boundaries["source_row_decode"] == "NOT_AUTHORIZED"
    assert boundaries["g4_open"] is False


@pytest.mark.parametrize("years", [[2010, 2014], [2017, 2017], [2000, 2021], [2021, 2022]])
def test_decision_years_may_not_touch_g1_model_selection_or_g4(years: list[int]) -> None:
    contract = copy.deepcopy(_contract())
    contract["years"]["decision_years"] = years
    with pytest.raises(ValueError):
        n10.validate_contract(contract)


def test_the_window_is_matured_and_excludes_the_sealed_and_opened_fields() -> None:
    contract = _contract()
    first, last = contract["years"]["decision_years"]
    assert (first, last) == (2000, 2012)
    assert last + 2 + 3 == contract["years"]["latest_outcome_year"] <= contract["source"]["last_complete_calendar_year"]
    for field in (27, 17):
        mutated = copy.deepcopy(contract)
        mutated["fields"]["ids"] = [16, 32, field]
        with pytest.raises(ValueError, match="excluded"):
            n10.validate_contract(mutated)


def test_the_fine_clock_threshold_is_the_n08_registered_one() -> None:
    contract = copy.deepcopy(_contract())
    assert contract["threshold_transfer"]["fine_clock_threshold"] == 0.098417
    contract["threshold_transfer"]["fine_clock_threshold"] = 0.09
    with pytest.raises(ValueError, match="N08 registered"):
        n10.validate_contract(contract)


def test_empirical_stages_each_need_their_own_authorization() -> None:
    stages = {stage["id"]: stage for stage in _contract()["stages"]}
    assert stages["T"]["empirical_rows"] == 0
    assert stages["S1"]["persistence_labels"] == 0
    for stage_id in ("S1", "S2", "S3"):
        assert stages[stage_id]["authorization"].startswith("SEPARATE_USER_AUTHORIZATION")
    assert stages["S2"]["may_change_registered_definitions"] is False


def test_the_statistic_matches_the_synthetic_definition() -> None:
    assert n10.spearman([1, 2], [0, 1]) is None
    assert n10.spearman([1, 1, 1], [0, 1, 1]) is None
    assert n10.spearman([1, 2, 3], [1, 1, 1]) is None
    assert n10.spearman([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1]) == pytest.approx((2 / 3) ** 0.5, abs=1e-12)


def test_cohort_pairs_and_the_equal_width_coarsening() -> None:
    steps, persisted = n10.cohort_pairs([[3, 2, 1], [9, 1, 0]])
    assert (steps, persisted) == ([3, 3, 9], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        n10.cohort_pairs([[3, 1, 2]])
    bins = [n10.coarsen(step, 3, 20, 3) for step in range(3, 21)]
    assert bins == [0] * 6 + [1] * 6 + [2] * 6
    for step in (2, 21):
        with pytest.raises(ValueError):
            n10.coarsen(step, 3, 20, 3)


def test_run_statistics_flags_a_reconstruction_mismatch() -> None:
    cohorts = [[3, 4, 1], [10, 4, 2], [16, 4, 3]]
    steps, persisted = n10.cohort_pairs(cohorts)
    geometry = {"steps": 24, "first": 3, "last": 20, "horizon": 4}
    good = {"cohort_persistence": cohorts, "outcome_selection_trend": n10.spearman(steps, persisted)}
    assert n10.run_statistics(good, geometry, 3, 1e-12)["reconstructed"] is True
    bad = {**good, "outcome_selection_trend": good["outcome_selection_trend"] + 1e-6}
    assert n10.run_statistics(bad, geometry, 3, 1e-12)["reconstructed"] is False


def _cells(values: list[tuple[str, float, float]]) -> list[dict]:
    return [
        {"study": "X", "cell_id": f"c{i}", "visibility_id": "v", "status": status, "coarse_value": value, "coarse_control": control}
        for i, (status, value, control) in enumerate(values)
    ]


def test_the_transfer_rule_takes_the_larger_threshold_and_checks_the_control() -> None:
    transfer = _contract()["threshold_transfer"]
    studies = [{"reconstruction_failures": 0, "cells": _cells([
        ("ENTRY_CHANNEL_ACTIVE", 0.30, -0.05), ("ENTRY_CHANNEL_ABSENT", 0.12, -0.02), ("ENTRY_SATURATED", 0.01, 0.0),
    ])}]
    result = n10.transfer_disposition(studies, transfer)
    assert result["disposition"] == "THRESHOLD_TRANSFERRED"
    assert result["coarse_threshold"] == 0.12
    assert result["data_threshold"] == 0.12
    studies[0]["cells"][1]["coarse_value"] = 0.05
    assert n10.transfer_disposition(studies, transfer)["data_threshold"] == 0.098417
    studies[0]["cells"][0]["coarse_control"] = 0.2
    assert n10.transfer_disposition(studies, transfer)["disposition"] == "THRESHOLD_NOT_TRANSFERABLE_CONTROL"
    studies[0]["reconstruction_failures"] = 1
    assert n10.transfer_disposition(studies, transfer)["disposition"] == "TRANSFER_RECONSTRUCTION_FAILED"


def test_the_surge_rule_uses_the_modified_z_and_both_count_conditions() -> None:
    rows = [{"topic_id": f"T{i}", "recent": [10 + i % 3] * 3, "reference": [10] * 3} for i in range(9)]
    rows.append({"topic_id": "SURGE", "recent": [60] * 3, "reference": [10] * 3})
    rows.append({"topic_id": "NEWBORN", "recent": [50] * 3, "reference": [0] * 3})
    result = n10.surge_flags(rows)
    assert result["surges"] == ["SURGE"]
    assert "NEWBORN" not in result["z"]
    flat = [{"topic_id": f"T{i}", "recent": [10] * 3, "reference": [10] * 3} for i in range(5)]
    assert n10.surge_flags(flat)["status"] == "NO_SURGE_MAD_ZERO_OR_NONFINITE"


def test_episode_selection_keeps_response_cohorts_apart() -> None:
    assert n10.select_episodes({"A": [2004, 2000, 2001, 2003, 2008]}) == {"A": [2000, 2003, 2008]}


def test_the_partition_is_deterministic_atomic_and_one_third_discovery() -> None:
    families = {16: [f"s{i}" for i in range(10)], 32: ["a", "b"], 11: ["x"]}
    seed = _contract()["partition"]["seed"]
    first = n10.partition_families(families, seed)
    reordered = n10.partition_families({k: list(reversed(v)) for k, v in reversed(list(families.items()))}, seed)
    assert first == reordered
    discovery = {f: sum(1 for (field, _s), part in first.items() if field == f and part == "discovery") for f in families}
    assert discovery == {16: 4, 32: 1, 11: 1}


def test_episode_floors() -> None:
    validity = _contract()["episode_validity"]
    pairs = [(k, p) for k, p in itertools.product((0, 1, 2), (0, 1))] * 5
    assert n10.episode_statistic(pairs, validity)["valid"] is True
    assert n10.episode_statistic(pairs[:20], validity)["meets_floor"] is False
    assert n10.episode_statistic([(0, 0), (0, 1)] * 20, validity)["meets_floor"] is False


def test_the_decision_rule() -> None:
    contract = _contract()
    strong = {16: [0.3, 0.25, 0.35, 0.28], 32: [0.22, 0.3, 0.26, 0.31], 11: [0.27, 0.33, 0.24, 0.29]}
    assert n10.decide(strong, 0.1, contract)["disposition"] == "INDICATOR_POSITIVE"
    assert n10.decide(strong, None, contract)["disposition"] == "DESCRIPTIVE_ONLY_THRESHOLD_NOT_TRANSFERRED"
    weak = {f: [v - 0.25 for v in values] for f, values in strong.items()}
    result = n10.decide(weak, 0.1, contract)
    assert result["disposition"] == "NOT_POSITIVE_NO_INFERENCE"
    decline = {f: [v - 0.5 for v in values] for f, values in strong.items()}
    assert n10.decide(decline, 0.1, contract)["later_cohort_decline_flag"] is True
    short = {**strong, 32: strong[32][:3]}
    assert n10.decide(short, 0.1, contract)["disposition"] == "INVALID_FIELD_SUPPORT"
    assert n10.decide(strong, 0.1, contract) == n10.decide(strong, 0.1, contract)


def test_the_precision_projection() -> None:
    contract = _contract()
    discovery = {16: [0.1, 0.3, 0.2], 32: [0.0, 0.4], 11: [0.2, 0.25, 0.15]}
    projection = n10.precision_projection(discovery, {16: 30, 32: 30, 11: 30}, 0.04, contract)
    assert projection["disposition"] == "GO"
    assert projection["field_sd"][32] == pytest.approx((0.105 / 5) ** 0.5)
    assert projection["projected_half_width"] == pytest.approx(1.645 * (0.0335 / 30) ** 0.5 / 3)
    assert n10.precision_projection(discovery, {16: 4, 32: 4, 11: 4}, 0.04, contract)["disposition"] == "STOP_INSUFFICIENT_PRECISION"
    assert n10.precision_projection(discovery, {16: 30, 32: 3, 11: 30}, 0.1, contract)["disposition"] == "STOP_INSUFFICIENT_EPISODES"
    assert n10.precision_projection({16: [0.1], 32: [0.2]}, {16: 30, 32: 30, 11: 30}, 0.1, contract)["disposition"] == "STOP_DISCOVERY_SUPPORT"


def test_the_module_has_no_empirical_row_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"pyarrow", "pandas", "polars", "duckdb"}
    assert "pivot-feedback-lean-h3-openalex-full-core" not in source


@pytest.mark.skipif(
    not Path("/ssd/pivot_observable_one_sided_v1_c0a8b1a_development/runs.jsonl").exists(),
    reason="the synthetic runs are machine-local",
)
def test_the_stored_runs_reconstruct_the_fine_clock_statistic() -> None:
    contract = _contract()
    transfer = contract["threshold_transfer"]
    for spec in transfer["studies"]:
        study_contract = n10.yaml.safe_load((ROOT / spec["contract"]).read_text(encoding="utf-8"))
        geometry = n10.study_geometry(study_contract, transfer["profile"])
        assert (geometry["first"], geometry["last"]) == (3, 20)
        with Path(spec["runs"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in itertools.islice(handle, 200)]
        assert all(n10.run_statistics(row, geometry, 3, 1e-12)["reconstructed"] for row in rows)
