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


def _contracts() -> tuple[dict, dict, dict]:
    contract = unclipped.load_contract()
    return contract, unclipped.load_llm_contract(contract), unclipped.load_v2_contract(contract)


def _saturated_log_odds(llm_contract: dict) -> dict[str, np.ndarray]:
    """Answers far below the clip that still vary with credibility, as P0/A did."""

    out = {}
    for regime in ("A", "B"):
        features, _ = llm.p0_cohort(llm_contract, regime)
        out[f"P0/{regime}"] = -16.0 + 4.0 * features[:, 0] - 2.0 * features[:, 2]
        out[f"X0/{regime}"] = np.zeros(len(llm.x0_keys(llm_contract, regime)))
    return out


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract, _llm, _v2 = _contracts()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(unclipped.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-unclipped-prior-v1.md",
        "experiments/configs/info_spread_policy_ladder_unclipped_prior_v1.yaml",
        "scripts/run_info_spread_policy_ladder_unclipped_prior_v1.py",
        "tests/test_info_spread_policy_ladder_unclipped_prior_v1.py",
    }
    assert contract["prior"]["llm_calls"] == 0


def test_without_the_clip_the_saturated_answers_keep_their_slope() -> None:
    _contract, llm_contract, _v2 = _contracts()
    log_odds = _saturated_log_odds(llm_contract)
    clipped = llm.build_llm_arms(llm_contract, log_odds)["report"]["A"]
    ours = unclipped.build_unclipped_priors(llm_contract, log_odds)
    report = ours["report"]["A"]
    assert clipped["p0_distillation_r_squared"] < 0.05  # the clip flattens them, as in D5
    assert report["p0u_distillation_r_squared"] > 0.999
    assert report["p0u_weights"][0] == pytest.approx(4.0, abs=0.05)
    assert report["p0u_weights"][2] == pytest.approx(-2.0, abs=0.05)
    features, _ = llm.p0_cohort(llm_contract, "A")
    assert ours["priors"]["A"].probability(features).mean() == pytest.approx(0.4, abs=1e-9)


def test_only_d5s_registered_elicitation_is_accepted(tmp_path: Path) -> None:
    contract, llm_contract, _v2 = _contracts()
    log_odds = _saturated_log_odds(llm_contract)
    records = [{"set": name, "index": i, "prompt_sha256": llm._text_sha256(text), "log_odds": float(log_odds[name][i]),
                "answer_token_mass": 1.0, "top_token_is_answer": True}
               for name, texts in llm.prompt_sets(llm_contract).items() for i, text in enumerate(texts)]
    directory = tmp_path / "stub"
    llm.write_elicitation(directory, {"records": records, "receipt": {}}, llm_contract, llm.DEFAULT_CONTRACT, None)
    with pytest.raises(RuntimeError):
        unclipped.load_registered_elicitation(contract, llm_contract, directory)


def test_a_smoke_study_runs_deterministically_and_passes_parity() -> None:
    contract, llm_contract, v2_contract = _contracts()
    v2.apply_learner_settings(v2_contract)
    log_odds = _saturated_log_odds(llm_contract)
    priors = unclipped.build_unclipped_priors(llm_contract, log_odds)["priors"]
    d5_arms = llm.build_llm_arms(llm_contract, log_odds)
    recorded = {"v2": {}, "d5_fixed": {}, "d5_learner": {}}
    cells = []
    for regime in v2_contract["design"]["regimes"]:
        first = unclipped.run_fixed_cell(v2_contract, priors, "smoke", regime, "linear", 0)
        assert json.dumps(first, sort_keys=True) == json.dumps(
            unclipped.run_fixed_cell(v2_contract, priors, "smoke", regime, "linear", 0), sort_keys=True)
        cells.append(first)
        recorded["d5_fixed"][(regime, "linear", 0)] = llm.run_fixed_cell(
            llm_contract, v2_contract, d5_arms, "smoke", regime, "linear", 0)
        for family in v2_contract["design"]["families"][regime]:
            cells.append(unclipped.run_learner_cell(v2_contract, priors, "smoke", regime, family, "linear", 0, "cpu"))
            recorded["v2"][(regime, family, "linear", 0)] = v2.run_cell(v2_contract, "smoke", regime, family, "linear",
                                                                        0, "cpu")
            recorded["d5_learner"][(regime, family, "linear", 0)] = llm.run_learner_cell(
                llm_contract, v2_contract, d5_arms, "smoke", regime, family, "linear", 0, "cpu")
    assert "seconds" not in json.dumps(cells)
    summary = unclipped.summarize(contract, v2_contract, recorded, cells)
    assert summary["parity_failures"] == []
    entry = summary["table"]["A/mlp/linear"]
    assert set(entry[v2.PRIMARY]) == {*unclipped.COMPARISONS, *unclipped.DESCRIPTIVE}
