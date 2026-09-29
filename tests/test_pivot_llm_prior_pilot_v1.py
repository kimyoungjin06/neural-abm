from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_llm_prior_pilot_v1 as pilot  # noqa: E402


FAKE_PRIOR = {"weights": [0.5, 2.0, 1.5, -2.0, -1.0, 0.5, 0.0, 0.0, 0.0], "bias": -0.5}


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_plan_is_local_and_renders_every_profile() -> None:
    contract = pilot.load_contract(CONFIG)
    plan = pilot._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert plan["profiles"] == 1024
    assert len(plan["source_bindings"]) == 10
    assert contract["boundaries"]["paid_api_calls"] == 0
    assert contract["llm"]["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert contract["elicitation"]["answer_tokens"] == ["Yes", "No"]
    assert "Answer with Yes or No." in plan["first_primary_prompt"]


def test_paid_api_and_reused_seeds_are_rejected() -> None:
    contract = _raw()
    contract["boundaries"]["paid_api_calls"] = 144
    with pytest.raises(ValueError, match="no paid API"):
        pilot.validate_contract(contract)
    contract = _raw()
    contract["probe_cohort"]["base_seed"] = 20261015
    with pytest.raises(ValueError, match="fresh seed"):
        pilot.validate_contract(contract)


def test_design_cohort_is_deterministic_and_prompts_carry_the_features() -> None:
    contract = pilot.load_contract(CONFIG)
    features, keys = pilot.design_cohort(contract)
    again, _keys = pilot.design_cohort(contract)
    np.testing.assert_array_equal(features, again)
    assert features.shape == (1024, 9)
    assert keys[0] == (0, 0) and keys[-1] == (3, 255)
    prompts = pilot.render_prompts(contract, features[:2])
    assert f"{features[1][1]:.2f}" in prompts["primary"][1]
    assert prompts["secondary"][0].endswith("Reply with only the number.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("72", 0.72), (" 35%", 0.35), ("100", 1.0), ("0", 0.0), ("probably", None), ("150", None)],
)
def test_verbalized_probability_parsing(text: str, expected: float | None) -> None:
    assert pilot.parse_verbalized(text) == expected


def test_distillation_recovers_a_known_logistic_prior() -> None:
    rng = np.random.default_rng(0)
    features = rng.random((1024, 9))
    weights = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.0, 0.25, 0.0])
    target = features @ weights + 0.3
    fitted = pilot.distill(features, target, ridge_per_profile=1e-9)
    np.testing.assert_allclose(fitted["weights"], weights, atol=1e-6)
    assert fitted["bias"] == pytest.approx(0.3, abs=1e-6)
    assert fitted["r_squared_on_log_odds"] == pytest.approx(1.0)
    probability = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(pilot.log_odds(probability)[1], 0.0)
    assert np.isfinite(pilot.log_odds(probability)).all()


def test_prior_slot_replaces_and_restores_even_on_error() -> None:
    features = np.random.default_rng(1).random((4, 9))
    hand = pilot.abm1.v2._base_logit(features)
    with pilot.prior_slot(FAKE_PRIOR):
        swapped = pilot.abm1.v2._base_logit(features)
        agent_weight = pilot.study2._initial_policy()[0]
    np.testing.assert_allclose(
        swapped, features @ np.array(FAKE_PRIOR["weights"]) + FAKE_PRIOR["bias"]
    )
    assert agent_weight.requires_grad
    assert pilot.study2._initial_policy is pilot.HAND_INITIAL_POLICY
    np.testing.assert_array_equal(pilot.abm1.v2._base_logit(features), hand)
    with pytest.raises(ZeroDivisionError), pilot.prior_slot(FAKE_PRIOR):
        _ = 1 / 0
    assert pilot.study2._initial_policy is pilot.HAND_INITIAL_POLICY


def test_prior_for_each_prior_and_world() -> None:
    oracles = {world: {"weights": [float(i)] * 9, "bias": 1.0} for i, world in enumerate(pilot.FIT_WORLDS)}
    assert pilot.prior_for("hand", "linear_fit", FAKE_PRIOR, oracles) is None
    assert pilot.prior_for("llm", "axis_interaction_fit", FAKE_PRIOR, oracles)["bias"] == -0.5
    assert pilot.prior_for("oracle_linear", "axis_interaction_fit", FAKE_PRIOR, oracles) == (
        oracles["axis_interaction_fit"]
    )
    assert pilot.prior_for("flat", "linear_fit", FAKE_PRIOR, oracles)["weights"] == [0.0] * 9


def test_oracle_priors_are_deterministic_and_world_specific() -> None:
    contract = pilot.load_contract(CONFIG)
    first = pilot.oracle_priors(contract)
    assert first == pilot.oracle_priors(contract)
    assert first["linear_fit"] != first["axis_interaction_fit"]
    # The linear world rewards resource security and penalizes skill distance.
    linear = first["linear_fit"]["weights"]
    index = {name: i for i, name in enumerate(pilot.FEATURES)}
    assert linear[index["resource_security"]] > 0 > linear[index["skill_distance"]]


def test_hand_task_nests_the_predecessor_and_llm_prior_changes_behavior() -> None:
    contract = pilot.load_contract(CONFIG)
    hand = pilot.run_task(contract, "smoke", 0, "linear_fit", "hand", None)
    swapped = pilot.run_task(contract, "smoke", 0, "linear_fit", "llm", FAKE_PRIOR)
    assert [row["arm_id"] for row in hand] == [arm for arm, _ in pilot.ABM_ARMS]
    assert hand[0]["probe_brier"] != swapped[0]["probe_brier"]
    assert pilot.study2._initial_policy is pilot.HAND_INITIAL_POLICY


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = pilot.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    serial, oracles = pilot.run_abm(reduced, "smoke", FAKE_PRIOR, workers=1)
    parallel, _oracles = pilot.run_abm(reduced, "smoke", FAKE_PRIOR, workers=3)
    assert len(serial) == 3 * len(pilot.PRIOR_IDS) * len(pilot.ABM_ARMS)
    assert serial == parallel
    assert set(oracles) == set(pilot.FIT_WORLDS)


def _rows(overrides: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in pilot.FIT_WORLDS:
            for prior in pilot.PRIOR_IDS:
                for arm, evidence in pilot.ABM_ARMS:
                    base = overrides.get((world, prior, arm), {})
                    rows.append(
                        {
                            "replicate": replicate,
                            "world_id": world,
                            "prior_id": prior,
                            "arm_id": arm,
                            "evidence_regime": evidence,
                            "probe_brier": base.get("probe_brier", 0.20) + wobble,
                            "productive_share_of_entrants": base.get("share", 0.30) + wobble,
                        }
                    )
    return rows


def test_decision_reads_quality_rescue_washout_and_entry() -> None:
    contract = pilot.load_contract(CONFIG)
    overrides = {
        ("linear_fit", "llm", "fixed_rule"): {"probe_brier": 0.15, "share": 0.36},
        ("linear_fit", "llm", "online_logistic"): {"probe_brier": 0.16},
        ("axis_interaction_fit", "llm", "fixed_rule"): {"probe_brier": 0.24},
        ("axis_interaction_fit", "llm", "tree_batch"): {"probe_brier": 0.23},
    }
    decision = pilot.decide(_rows(overrides), contract)["by_world"]
    linear = decision["linear_fit"]
    assert linear["prior_quality"] == "LLM_PRIOR_BETTER"
    assert linear["local_rescue"] == "LLM_PRIOR_IMPROVES_LOCAL_LEARNING"
    assert linear["pooled_washout"] == "PRIOR_WASHES_OUT_UNDER_POOLING"
    assert linear["entry_precision"] == "LLM_PRIOR_SELECTS_BETTER_ENTRANTS"
    axis = decision["axis_interaction_fit"]
    assert axis["prior_quality"] == "HAND_PRIOR_BETTER"
    assert axis["pooled_washout"] == "PRIOR_PERSISTS_UNDER_POOLING"
    assert axis["prior_persists_for_arms"] == ["tree_batch"]
    assert decision["smooth_interaction_fit"]["prior_quality"] == "NO_CLEAR_DIFFERENCE"


def _fake_elicitation(directory: Path, commit: str) -> None:
    directory.mkdir()
    (directory / "elicitation.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / "llm_prior.json").write_text(json.dumps(FAKE_PRIOR), encoding="utf-8")
    receipt = {
        "precommit": {"commit_sha": commit},
        "outputs": {
            name: pilot._sha256(directory / name)
            for name in ("elicitation.jsonl", "llm_prior.json")
        },
    }
    (directory / "elicitation_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_elicitation_is_bound_to_the_precommit_and_its_hashes(tmp_path: Path) -> None:
    commit = "a" * 40
    _fake_elicitation(tmp_path / "e", commit)
    assert pilot.load_elicitation(tmp_path / "e", commit)["bias"] == -0.5
    with pytest.raises(RuntimeError, match="different precommit"):
        pilot.load_elicitation(tmp_path / "e", "b" * 40)
    (tmp_path / "e" / "llm_prior.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        pilot.load_elicitation(tmp_path / "e", commit)


def test_development_summary_and_outputs_write_from_full_shaped_rows(
    tmp_path: Path,
) -> None:
    contract = pilot.load_contract(CONFIG)
    template = pilot.run_task(contract, "smoke", 0, "linear_fit", "hand", None)
    rows = [
        {**row, "replicate": replicate, "world_id": world, "prior_id": prior}
        for replicate in range(16)
        for world in pilot.FIT_WORLDS
        for prior in pilot.PRIOR_IDS
        for row in template
    ]
    oracles = {world: FAKE_PRIOR for world in pilot.FIT_WORLDS}
    summary = pilot.summarize(rows, contract, "development", oracles, FAKE_PRIOR)
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["cells"]) == 3 * 4 * 5
    _fake_elicitation(tmp_path / "e", "a" * 40)
    receipt = pilot.write_abm(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "a" * 40},
        contract_path=CONFIG,
        elicitation_dir=tmp_path / "e",
    )
    assert receipt["row_count"] == 960
    assert set(receipt["elicitation"]) == {
        "elicitation.jsonl",
        "llm_prior.json",
        "elicitation_receipt.json",
    }


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = pilot.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        pilot.delayed.verify_precommit("short", contract)
