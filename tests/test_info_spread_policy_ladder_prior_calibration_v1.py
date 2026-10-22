from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_prior_calibration_v1 as study  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_unclipped_prior_v1 as unclipped  # noqa: E402
import run_info_spread_policy_ladder_utility_v1 as o25  # noqa: E402


def _contracts() -> tuple[dict, dict, dict, dict]:
    contract = study.load_contract()
    return (contract, *study.contracts(contract))


def _stub_log_odds(llm_contract: dict) -> dict[str, np.ndarray]:
    out = {}
    for regime in ("A", "B"):
        features, _ = llm.p0_cohort(llm_contract, regime)
        out[f"P0/{regime}"] = -12.0 + 3.0 * features[:, 0] - 2.0 * features[:, 2]
        keys = llm.x0_keys(llm_contract, regime)
        out[f"X0/{regime}"] = np.asarray([0.2 * a - 0.1 * c for a, c, _s in keys])
    return out


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract, *_ = _contracts()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(study.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-prior-calibration-v1.md",
        "experiments/configs/info_spread_policy_ladder_prior_calibration_v1.yaml",
        "scripts/run_info_spread_policy_ladder_prior_calibration_v1.py",
        "tests/test_info_spread_policy_ladder_prior_calibration_v1.py",
    }
    assert contract["dispositions"]["costed_cases"] == ["P0/A", "P0/B", "P0u/B"]


def test_each_comparison_pairs_an_llm_component_against_its_hand_counterpart() -> None:
    # The two routes from L0 to L1 add up: calibration then answers, or answers then calibration.
    for named in study.COMPARISONS.values():
        cal_llm, cal_hand = named["calibration_given_llm_answers"], named["calibration_given_hand_answers"]
        ans_hand, ans_llm = named["answers_given_hand_calibration"], named["answers_given_llm_calibration"]
        assert cal_llm[1] == ans_hand[0] and ans_hand[1] == cal_hand[1] == "L0"
        assert cal_hand[0] == ans_llm[1] and cal_llm[0] == ans_llm[0]


def test_recalibration_keeps_the_direction_and_takes_the_location_and_steepness() -> None:
    _contract, _v2, llm_contract, _o24 = _contracts()
    cohort, _ = llm.p0_cohort(llm_contract, "B")
    rng = np.random.default_rng(3)
    direction = core.LinearPrior(rng.normal(size=cohort.shape[1]), 1.5)
    source = core.LinearPrior(rng.normal(size=cohort.shape[1]) * 4.0, -2.0)
    made = study.recalibrate(direction, source, cohort, llm_contract["calibration"])
    ratio = made.weight / direction.weight
    assert np.all(ratio > 0) and np.allclose(ratio, ratio[0])
    ours, theirs = study.cohort_calibration(made, cohort), study.cohort_calibration(source, cohort)
    assert ours["mean_probability"] == pytest.approx(theirs["mean_probability"], abs=1e-9)
    assert ours["logit_sd"] == pytest.approx(theirs["logit_sd"], rel=1e-12)
    same = study.recalibrate(source, source, cohort, llm_contract["calibration"])
    assert np.array_equal(same.weight, source.weight) and same.bias == pytest.approx(source.bias, abs=1e-6)


def test_the_built_priors_match_their_calibration_sources_on_the_cohort() -> None:
    _contract, _v2, llm_contract, _o24 = _contracts()
    report = study.build_priors(llm_contract, _stub_log_odds(llm_contract))["report"]
    for regime in ("A", "B"):
        assert set(report[regime]) == {"F1", "P0", "P0u", *study.PRIORS}
        for name, (_direction, source) in study.PRIORS.items():
            for key in ("mean_probability", "logit_sd"):
                assert report[regime][name][key] == pytest.approx(report[regime][source][key], abs=2e-6), name
        assert report[regime]["F1"]["mean_probability"] > 0.6
        assert report[regime]["P0"]["mean_probability"] == pytest.approx(0.4, abs=1e-6)


def _smoke(contract: dict, v2_contract: dict, llm_contract: dict) -> tuple[dict, dict, list]:
    v2.apply_learner_settings(v2_contract)
    log_odds = _stub_log_odds(llm_contract)
    d5_arms = llm.build_llm_arms(llm_contract, log_odds)
    o25_arms = o25.priors_and_tables(llm_contract, log_odds)
    priors = study.build_priors(llm_contract, log_odds)["priors"]
    cells: dict[str, list] = {"study_v2": [], "llm_arms": [], "unclipped_prior": [], "utility": []}
    ours = []
    for regime in v2_contract["design"]["regimes"]:
        cells["llm_arms"].append(llm.run_fixed_cell(llm_contract, v2_contract, d5_arms, "smoke", regime, "linear", 0))
        cells["unclipped_prior"].append(unclipped.run_fixed_cell(v2_contract, o25_arms["P0u"], "smoke", regime,
                                                                 "linear", 0))
        ours.append(study.run_fixed_cell(v2_contract, priors, "smoke", regime, "linear", 0))
        if regime == "B":
            cells["utility"].append(o25.run_fixed_cell(llm_contract, v2_contract, o25_arms, "smoke", regime, "linear", 0))
        for family in v2_contract["design"]["families"][regime]:
            cells["study_v2"].append(v2.run_cell(v2_contract, "smoke", regime, family, "linear", 0, "cpu"))
            cells["llm_arms"].append(llm.run_learner_cell(llm_contract, v2_contract, d5_arms, "smoke", regime, family,
                                                          "linear", 0, "cpu"))
            cells["unclipped_prior"].append(unclipped.run_learner_cell(v2_contract, o25_arms["P0u"], "smoke", regime,
                                                                       family, "linear", 0, "cpu"))
            if regime == "B":
                cells["utility"].append(o25.run_learner_cell(v2_contract, o25_arms, "smoke", regime, family, "linear",
                                                             0, "cpu"))
            ours.append(study.run_learner_cell(v2_contract, priors, "smoke", regime, family, "linear", 0, "cpu"))
    contract = {**contract, "profile_for_counts": "smoke"}
    recorded = study.recorded_values(contract, v2_contract, cells)
    # O25's record, rebuilt from the same smoke cells.
    o25_contract = {**o25.load_contract(), "profile_for_counts": "smoke"}
    joined = o25.join_recorded(cells["study_v2"], cells["llm_arms"], cells["unclipped_prior"])
    values, failures = o25.utility_rows(o25_contract, v2_contract, joined, cells["utility"])
    assert failures == []
    record = o25.summarize(o25_contract, v2_contract, values)
    utility = {"A": "MOSTLY_NEGATIVE", "B": "MOSTLY_NEGATIVE"}
    record["dispositions"] = {"by_comparison": {
        **{f"llm_prior_in_learner/{r}": {"utility": u} for r, u in utility.items()},
        "unclipped_prior_in_learner/A": {"utility": "MIXED_OR_NONE"},
        "unclipped_prior_in_learner/B": {"utility": "MOSTLY_NEGATIVE"},
    }}
    return contract, record, [recorded, ours]


def test_a_smoke_study_joins_the_records_with_parity_and_consistency() -> None:
    contract, v2_contract, llm_contract, _o24 = _contracts()
    contract, record, (recorded, ours) = _smoke(contract, v2_contract, llm_contract)
    summary = study.summarize(contract, v2_contract, record, recorded, ours)
    assert summary["parity_failures"] == []
    assert summary["consistency_failures"] == []
    assert set(summary["calibration_check"]) == {f"{p}/{r}" for p in study.PRIORS for r in ("A", "B")}
    group = summary["table"]["A/mlp/linear"]
    assert set(study.flat_comparisons()) <= set(group)
    assert set(group["mean_utility"]) == {"L0", "L1", "L1u", *study.LEARNERS}
    assert summary["dispositions"]["costed_cases"] == ["P0/A", "P0/B", "P0u/B"]
    assert summary["dispositions"]["overall"].startswith("O28_")
    # A recorded L0 that differs from the recomputation fails parity.
    broken = copy.deepcopy(recorded)
    broken[("B", "partial_mlp", "linear", 0)][0]["arms"]["L0"]["parity"]["decisions"] += 1
    assert study.parity_failures(broken, ours) == ["B/partial_mlp/linear/0"]


def _table_with(labels: dict[str, str], regime: str = "A") -> dict:
    sign = {"NEGATIVE": -1.0, "POSITIVE": 1.0, "INDISTINGUISHABLE": 0.0}
    table = {}
    for index, geometry in enumerate(("linear", "smooth", "axis")):
        for family in ("f1", "f2"):
            entry = {}
            for name in study.flat_comparisons():
                prior, comparison = name.split("/", 1)
                label = labels.get(comparison, "INDISTINGUISHABLE") if prior == "P0" else "INDISTINGUISHABLE"
                entry[name] = {"material": label != "INDISTINGUISHABLE", "mean_cell_delta": sign[label] * (1 + index)}
            table[f"{regime}/{family}/{geometry}"] = entry
    return table


@pytest.mark.parametrize(("labels", "expected"), [
    ({"calibration_given_llm_answers": "NEGATIVE", "calibration_given_hand_answers": "NEGATIVE"}, "CALIBRATION"),
    ({"answers_given_hand_calibration": "NEGATIVE", "answers_given_llm_calibration": "NEGATIVE"}, "ANSWERS"),
    ({name: "NEGATIVE" for name in (*study.CALIBRATION, *study.ANSWERS)}, "BOTH"),
    ({"calibration_given_llm_answers": "NEGATIVE"}, "NEITHER_CONSISTENTLY"),
])
def test_attribution_reads_both_routes(labels: dict, expected: str) -> None:
    contract, v2_contract, _llm, _o24 = _contracts()
    contract = {**contract, "dispositions": {**contract["dispositions"], "costed_cases": ["P0/A"]}}
    record = {"dispositions": {"by_comparison": {"llm_prior_in_learner/A": {"utility": "MOSTLY_NEGATIVE"},
                                                 "llm_prior_in_learner/B": {"utility": "MIXED_OR_NONE"},
                                                 "unclipped_prior_in_learner/A": {"utility": "MIXED_OR_NONE"},
                                                 "unclipped_prior_in_learner/B": {"utility": "MIXED_OR_NONE"}}}}
    check = {f"{p}/{r}": {"moves_to_its_calibration_source": True} for p in study.PRIORS for r in ("A", "B")}
    result = study.dispositions(contract, v2_contract, record, _table_with(labels), check)
    assert result["by_case"]["P0/A"]["attribution"] == expected
    assert result["overall"] == f"O28_{expected}_IN_EVERY_COSTED_CASE"
    check["F1_pcal/A"]["moves_to_its_calibration_source"] = False
    failed = study.dispositions(contract, v2_contract, record, _table_with(labels), check)
    assert failed["by_case"]["P0/A"]["attribution"] == "CALIBRATION_CHECK_FAILED"
    assert failed["overall"] == "O28_CALIBRATION_CHECK_FAILED"
