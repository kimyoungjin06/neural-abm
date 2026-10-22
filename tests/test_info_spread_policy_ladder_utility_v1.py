from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402
import run_info_spread_policy_ladder_unclipped_prior_v1 as unclipped  # noqa: E402
import run_info_spread_policy_ladder_utility_v1 as study  # noqa: E402


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
        "docs/research/info-spread-policy-ladder-utility-v1.md",
        "experiments/configs/info_spread_policy_ladder_utility_v1.yaml",
        "scripts/run_info_spread_policy_ladder_utility_v1.py",
        "tests/test_info_spread_policy_ladder_utility_v1.py",
        "scripts/info_spread_utility_measure_v1.py",
        "tests/test_info_spread_utility_measure_v1.py",
    }


def test_the_group_rule_uses_a_utility_scale_minimum() -> None:
    material = study.group_rule([2.0, 2.2, 1.9, 2.1], (1, 2), minimum=1.0, resamples=500)
    too_small = study.group_rule([0.2, 0.22, 0.19, 0.21], (1, 2), minimum=1.0, resamples=500)
    assert material["material"] and not too_small["material"]


def test_a_smoke_study_computes_utility_from_counts_in_a_and_reruns_b_with_parity() -> None:
    contract, v2_contract, llm_contract, _o24 = _contracts()
    v2.apply_learner_settings(v2_contract)
    log_odds = _stub_log_odds(llm_contract)
    d5_arms = llm.build_llm_arms(llm_contract, log_odds)
    arms = study.priors_and_tables(llm_contract, log_odds)
    priors_u = arms["P0u"]
    v2_cells, d5_cells, o24_cells, reruns = [], [], [], []
    for regime in v2_contract["design"]["regimes"]:
        d5_cells.append(llm.run_fixed_cell(llm_contract, v2_contract, d5_arms, "smoke", regime, "linear", 0))
        o24_cells.append(unclipped.run_fixed_cell(v2_contract, priors_u, "smoke", regime, "linear", 0))
        if regime == "B":
            reruns.append(study.run_fixed_cell(llm_contract, v2_contract, arms, "smoke", regime, "linear", 0))
        for family in v2_contract["design"]["families"][regime]:
            v2_cells.append(v2.run_cell(v2_contract, "smoke", regime, family, "linear", 0, "cpu"))
            d5_cells.append(llm.run_learner_cell(llm_contract, v2_contract, d5_arms, "smoke", regime, family,
                                                 "linear", 0, "cpu"))
            o24_cells.append(unclipped.run_learner_cell(v2_contract, priors_u, "smoke", regime, family, "linear",
                                                        0, "cpu"))
            if regime == "B":
                reruns.append(study.run_learner_cell(v2_contract, arms, "smoke", regime, family, "linear", 0, "cpu"))
    recorded = study.join_recorded(v2_cells, d5_cells, o24_cells)
    contract = {**contract, "profile_for_counts": "smoke"}
    values, parity_failures = study.utility_rows(contract, v2_contract, recorded, reruns)
    assert parity_failures == []
    assert set(values[("A", "mlp", "linear", 0)][0]) == set(study.ARMS)
    assert "seconds" not in json.dumps(reruns)
    summary = study.summarize(contract, v2_contract, values)
    assert set(summary["minimum_effect"]) == {"A", "B"}
    assert set(summary["table"]["B/partial_mlp/linear"]) >= set(study.COMPARISONS)


def test_a_rerun_that_differs_from_its_record_fails_parity() -> None:
    contract, v2_contract, _llm, _o24 = _contracts()
    metrics = {"share_precision": 0.5, "unverified_share_precision": 0.5, "false_reach": 1.0, "decisions": 10,
               "by_state": {"unrevealed": {"decisions": 1, "shares": 1, "true": 1}}}
    recorded = {("B", "mlp", "linear", 0): [{"replicate": 0, "arms": {"F1": metrics}}]}
    rerun = [{"kind": "learner", "regime": "B", "family": "mlp", "geometry": "linear", "cell": 0,
              "rows": [{"replicate": 0, "arms": {"F1": {**metrics, "decisions": 11,
                                                        study.UTILITY: 0.0}}}]},
             {"kind": "fixed", "regime": "B", "geometry": "linear", "cell": 0, "rows": [{"replicate": 0, "arms": {}}]}]
    _values, failures = study.utility_rows(contract, v2_contract, recorded, rerun)
    assert failures == ["B/mlp/linear/0/0/F1"]


@pytest.mark.parametrize(("precision", "utility", "agrees"), [
    ({"POSITIVE": 5, "NEGATIVE": 0, "INDISTINGUISHABLE": 1}, {"POSITIVE": 4, "NEGATIVE": 0, "INDISTINGUISHABLE": 2}, True),
    ({"POSITIVE": 5, "NEGATIVE": 0, "INDISTINGUISHABLE": 1}, {"POSITIVE": 2, "NEGATIVE": 0, "INDISTINGUISHABLE": 4}, False),
])
def test_agreement_compares_the_per_regime_summaries(precision: dict, utility: dict, agrees: bool,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    contract, v2_contract, _llm, _o24 = _contracts()
    calls = iter([utility, precision])

    def fake_counts(_table: dict, _v2: dict) -> dict:
        counts = next(calls)
        return {name: {regime: counts for regime in ("A", "B")} for name in study.COMPARISONS}

    monkeypatch.setattr(study, "label_counts", fake_counts)
    monkeypatch.setattr(study, "precision_tables", lambda _contract: {})
    result = study.dispositions(contract, v2_contract, {})
    assert all(row["agrees"] == agrees for row in result["by_comparison"].values())
    assert result["overall"] == ("CONCLUSIONS_HOLD_UNDER_UTILITY" if agrees else "SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY")
