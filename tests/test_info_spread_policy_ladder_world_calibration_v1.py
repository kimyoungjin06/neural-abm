from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_info_spread_policy_ladder_prior_calibration_v1 as o28  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_world_calibration_v1 as study  # noqa: E402
import test_info_spread_policy_ladder_prior_calibration_v1 as o28_tests  # noqa: E402


class SerialPool:
    def map(self, function, jobs):  # noqa: ANN001, ANN201
        return [function(job) for job in jobs]


def _contracts() -> tuple[dict, dict, dict, dict]:
    contract = study.load_contract()
    return (contract, *study.contracts(contract))


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract, *_ = _contracts()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(study.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-world-calibration-v1.md",
        "experiments/configs/info_spread_policy_ladder_world_calibration_v1.yaml",
        "scripts/run_info_spread_policy_ladder_world_calibration_v1.py",
        "tests/test_info_spread_policy_ladder_world_calibration_v1.py",
    }
    assert contract["dispositions"]["costed_cases"] == ["P0/A", "P0/B", "P0u/B"]
    # Each world-calibrated prior keeps an O28 prior's weights and plays its role in the same comparisons.
    assert {study.sources(n) for n in study.PRIORS} == set(o28.PRIORS.values())


def test_the_calibration_worlds_are_disjoint_from_the_study_cells() -> None:
    contract, v2_contract, _llm, _o24 = _contracts()
    runs = study.calibration_runs(contract, v2_contract)
    assert len(runs) == 24
    seeds = {study.calibration_world(contract, v2_contract, "smoke", "A", g, i).config.world_seed for g, i, _r in runs}
    study_seeds = {int(v2_contract["seeds"]["world_seed_offset"]) + c for c in range(20)}
    assert seeds == set(range(15000, 15012)) and not seeds & study_seeds


def _steps(bias: float) -> float:
    return float(math.floor(1000 * (1.0 / (1.0 + math.exp(-bias)))))


def test_locate_keeps_a_matching_cohort_location_and_otherwise_bisects_to_the_largest_match() -> None:
    contract, *_ = _contracts()
    spec = contract["world_calibration"]
    kept = study.locate(_steps, target=_steps(0.3) * 1.01, cohort_bias=0.3, spec=spec)
    assert kept["method"] == "KEPT_COHORT_LOCATION" and kept["bias"] == 0.3
    located = study.locate(_steps, target=700.0, cohort_bias=3.0, spec=spec)
    assert located["method"] == "BISECTED"
    assert _steps(located["bias"]) <= 700.0 < located["just_above"]
    # The largest intercept whose count stays at 700 sits just below sigmoid = 0.701.
    assert located["bias"] == pytest.approx(math.log(0.701 / 0.299), abs=1e-6)


def _small(contract: dict) -> dict:
    spec = {**contract["world_calibration"], "worlds_per_geometry": 1, "replicates": [0], "bisection_steps": 6}
    return {**contract, "world_calibration": spec}


def test_a_smoke_study_calibrates_in_the_world_and_joins_the_records(monkeypatch: pytest.MonkeyPatch) -> None:
    contract, v2_contract, llm_contract, _o24 = _contracts()
    o28_contract = o28.load_contract()
    o28_contract, record, (recorded, _ours) = o28_tests._smoke(o28_contract, v2_contract, llm_contract)
    contract = _small({**contract, "profile_for_counts": "smoke"})
    priors = o28.build_priors(llm_contract, o28_tests._stub_log_odds(llm_contract))["priors"]
    path = ROOT / "experiments" / "configs" / "info_spread_policy_ladder_world_calibration_v1.yaml"
    pool = SerialPool()

    def mean_shares(_pool, _path, _contract, v2c, profile, regime, prior):  # noqa: ANN001, ANN202
        runs = study.calibration_runs(contract, v2c)
        return float(np.mean([study.calibration_shares(contract, v2c, profile, regime, g, i, r, prior)
                              for g, i, r in runs]))

    monkeypatch.setattr(study, "mean_shares", mean_shares)
    calibrated = study.world_calibrate(pool, str(path), contract, v2_contract, priors, "smoke")
    assert set(calibrated["biases"]["A"]) == set(study.PRIORS)
    for regime in ("A", "B"):
        for name in study.PRIORS:
            row = calibrated["report"][regime][name]
            assert row["method"] in ("KEPT_COHORT_LOCATION", "BISECTED")
            if row["method"] == "BISECTED":
                assert row["achieved"] <= row["target"] < row["just_above"]
    world = study.world_priors(priors, calibrated["biases"])
    v2.apply_learner_settings(v2_contract)
    ours = []
    for regime in v2_contract["design"]["regimes"]:
        ours.append(study.run_fixed_cell(v2_contract, world, "smoke", regime, "linear", 0))
        for family in v2_contract["design"]["families"][regime]:
            ours.append(study.run_learner_cell(v2_contract, world, "smoke", regime, family, "linear", 0, "cpu"))
    o28_record = {"dispositions": {"by_case": {case: {"attribution": "CALIBRATION"}
                                               for case in ("P0/A", "P0/B", "P0u/A", "P0u/B")}}}
    summary = study.summarize(contract, v2_contract, {"utility": record, "prior_calibration": o28_record},
                              recorded, ours)
    assert summary["parity_failures"] == [] and summary["consistency_failures"] == []
    assert set(summary["world_match_check"]) == {f"{p}/{r}" for p in study.PRIORS for r in ("A", "B")}
    assert set(summary["table"]["A/mlp/linear"]["mean_utility"]) == {"L0", "L1", "L1u", *study.LEARNERS}
    assert summary["dispositions"]["overall"].startswith("O33_")


CAL = {"calibration_given_llm_answers": "NEGATIVE", "calibration_given_hand_answers": "NEGATIVE"}


@pytest.mark.parametrize(("p0", "p0u", "matched", "overall"), [
    (CAL, {}, True, "O33_C63_AND_C64_HOLD"),
    (CAL, CAL, True, "O33_ONLY_C63_HOLDS"),
    ({}, {}, True, "O33_ONLY_C64_HOLDS"),
    ({}, CAL, True, "O33_NEITHER_HOLDS"),
    (CAL, {}, False, "O33_WORLD_MATCH_FAILED"),
])
def test_the_overall_reads_which_claims_hold(p0: dict, p0u: dict, matched: bool, overall: str) -> None:
    contract, v2_contract, _llm, _o24 = _contracts()
    table = {}
    sign = {"NEGATIVE": -1.0, "INDISTINGUISHABLE": 0.0}
    for regime in ("A", "B"):
        for index, geometry in enumerate(("linear", "smooth", "axis")):
            for family in ("f1", "f2"):
                entry = {}
                for name in study.flat_comparisons():
                    prior, comparison = name.split("/", 1)
                    label = (p0 if prior == "P0" else p0u).get(comparison, "INDISTINGUISHABLE")
                    entry[name] = {"material": label != "INDISTINGUISHABLE",
                                   "mean_cell_delta": sign[label] * (1 + index)}
                table[f"{regime}/{family}/{geometry}"] = entry
    check = {f"{p}/{r}": {"matches": True} for p in study.PRIORS for r in ("A", "B")}
    if not matched:
        check["F1_pcalw/B"]["matches"] = False
    records = {"utility": {"dispositions": {"by_comparison": {
        "llm_prior_in_learner/A": {"utility": "MOSTLY_NEGATIVE"}, "llm_prior_in_learner/B": {"utility": "MOSTLY_NEGATIVE"},
        "unclipped_prior_in_learner/A": {"utility": "MIXED_OR_NONE"},
        "unclipped_prior_in_learner/B": {"utility": "MOSTLY_NEGATIVE"}}}},
        "prior_calibration": {"dispositions": {"by_case": {c: {"attribution": "CALIBRATION"}
                                                           for c in ("P0/A", "P0/B", "P0u/A", "P0u/B")}}}}
    result = study.dispositions(contract, v2_contract, records, table, check)
    assert result["overall"] == overall
    if matched:
        assert result["by_case"]["P0/A"]["o28_attribution"] == "CALIBRATION"
