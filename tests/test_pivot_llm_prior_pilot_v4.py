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
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v4.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_llm_prior_pilot_v4 as pilot  # noqa: E402


_RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
V1_DIR = Path(_RAW["v3_reference"]["v1_elicitation"]["directory"])
V2_PRIORS = Path(_RAW["v3_reference"]["v2_calibrated_priors"]["path"])
needs_predecessors = pytest.mark.skipif(
    not (V1_DIR.exists() and V2_PRIORS.exists()), reason="v1 and v2 artifacts are machine-local"
)
FAKE_PRIOR = {"weights": [0.5, 2.0, 1.5, -2.0, -1.0, 0.5, 0.0, 0.0, 0.0], "bias": -1.5}
FAKE_PRIORS = {source: dict(FAKE_PRIOR) for source in pilot.PRIOR_SOURCES}


def _raw() -> dict:
    return copy.deepcopy(_RAW)


def _fake_result(contract: dict, model: str, tilt: float) -> dict:
    features, keys = pilot.pilot1.design_cohort(contract)
    logit = features @ np.asarray(FAKE_PRIOR["weights"]) * tilt + FAKE_PRIOR["bias"]
    probability = 1.0 / (1.0 + np.exp(-logit))
    distilled = pilot.pilot1.distill(features, pilot.pilot1.log_odds(probability), 0.001)
    distilled.pop("fitted_log_odds")
    prompts = pilot.pilot1.render_prompts(contract, features)["primary"]
    records = [
        {
            "index": index,
            "replicate": key[0],
            "agent_id": key[1],
            "features": [round(float(value), 10) for value in row],
            "primary_prompt_sha256": pilot.pilot1._text_sha256(prompts[index]),
            "primary_log_odds": float(logit[index]),
            "primary_probability": float(probability[index]),
            "answer_token_mass": 1.0,
            "top_token_is_answer": True,
            "distilled_probability": float(probability[index]),
        }
        for index, (key, row) in enumerate(zip(keys, features, strict=True))
    ]
    validity = {"median_answer_token_mass": 1.0, "share_top_token_is_answer": 1.0, "weak_format": False}
    receipt = {
        "model": model,
        "parameters": 1,
        "prompts_identical_to_v1": True,
        "assistant_prefill": contract["elicitation"]["readout"]["assistant_prefill"],
        "validity": validity,
        "v1_first_token_readout_validity": validity,
        "determinism": {"repeat_pass_identical": True},
        "timing_seconds": {
            "primary_forward_total": 10.0,
            "primary_forward_per_prompt": 0.01,
            "primary_forward_per_prompt_in_sensitivity_batch": 0.002,
        },
        "peak_gpu_memory_bytes": 1,
    }
    prior = {"model": model, **distilled, "fidelity": {"r_squared_on_log_odds": 1.0}}
    return {"records": records, "llm_prior": prior, "receipt": receipt}


def _fake_root(tmp_path: Path, contract: dict, precommit_sha: str = "0" * 40) -> Path:
    root = tmp_path / "elicitation"
    for index, model in enumerate(pilot.MODEL_IDS):
        pilot.write_elicitation(
            root / model,
            _fake_result(contract, model, 1.0 + 0.1 * index),
            {"commit_sha": precommit_sha},
            CONFIG,
        )
    return root


def test_contract_plan_binds_sources_models_and_fixed_shares() -> None:
    contract = pilot.load_contract(CONFIG)
    plan = pilot._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 14
    assert plan["models"] == ["qwen3_0_6b", "qwen3_1_7b", "qwen3_4b", "qwen3_4b_2507"]
    epsilons = plan["epsilon_by_condition"]
    assert len(plan["conditions"]) == 12
    for condition in plan["conditions"]:
        if condition.startswith("hybrid_"):
            assert 1.0 - (1.0 - epsilons[condition]) ** 16 == pytest.approx(0.2)
        else:
            assert epsilons[condition] == 0.0


def test_contract_rejects_reused_seeds_other_prompts_tuned_shares_and_paid_calls() -> None:
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261023
    with pytest.raises(ValueError, match="fresh seed"):
        pilot.validate_contract(contract)
    contract = _raw()
    contract["elicitation"]["design_cohort"]["base_seed"] = 20261099
    with pytest.raises(ValueError, match="v1 cohort"):
        pilot.validate_contract(contract)
    contract = _raw()
    contract["conditions"][3]["exploratory_share"] = 0.1
    with pytest.raises(ValueError, match="exploratory share"):
        pilot.validate_contract(contract)
    contract = _raw()
    contract["boundaries"]["paid_api_calls"] = 1
    with pytest.raises(ValueError, match="paid API"):
        pilot.validate_contract(contract)


def test_chat_rendering_passes_the_model_template_kwargs() -> None:
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN202
            seen.append(kwargs)
            return messages[-1]["content"]

    rendered = pilot.render_chat(Tokenizer(), "system", ["a", "b"], {"enable_thinking": False})
    assert rendered == ["a", "b"]
    assert all(kwargs["enable_thinking"] is False for kwargs in seen)
    assert all(kwargs["add_generation_prompt"] is True for kwargs in seen)


@needs_predecessors
def test_v3_reference_prior_rebuilds_exactly_from_the_v1_elicitation() -> None:
    contract = pilot.load_contract(CONFIG)
    elicitation = pilot.load_elicitation(V1_DIR, None)
    rebuilt = pilot.calibrate(contract, elicitation["records"], elicitation["llm_prior"])
    stored = json.loads(V2_PRIORS.read_text(encoding="utf-8"))["priors"]["llm_base_rate"]
    assert rebuilt["weights"] == stored["weights"]
    assert rebuilt["bias"] == stored["bias"]
    assert len(pilot.load_v1_prompt_hashes(contract)) == 1024


def test_elicitation_loader_rejects_tampering_and_foreign_precommits(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    directory = tmp_path / "model"
    pilot.write_elicitation(
        directory, _fake_result(contract, "qwen3_0_6b", 1.0), {"commit_sha": "a" * 40}, CONFIG
    )
    assert pilot.load_elicitation(directory, "a" * 40)["receipt"]["model"] == "qwen3_0_6b"
    with pytest.raises(RuntimeError, match="different precommit"):
        pilot.load_elicitation(directory, "b" * 40)
    prior_path = directory / "llm_prior.json"
    prior_path.chmod(0o600)
    prior_path.write_text(prior_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        pilot.load_elicitation(directory, "a" * 40)


@needs_predecessors
def test_priors_calibrate_every_source_and_phase_a_reports_each(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    root = _fake_root(tmp_path, contract)
    elicitations = pilot.load_elicitations(contract, root, "0" * 40)
    priors = pilot.build_priors(contract, elicitations)
    assert list(priors) == list(pilot.PRIOR_SOURCES)
    for prior in priors.values():
        assert prior["achieved_mean"] == pytest.approx(0.30, abs=1e-9)
    profile = pilot.phase_a_profile(contract, elicitations, priors)
    assert set(profile["models"]) == set(pilot.PRIOR_SOURCES)
    assert profile["models"][pilot.REFERENCE]["readout"] == "V1_FIRST_TOKEN"
    assert profile["models"]["qwen3_4b_2507"]["spearman_readout_versus_anchor"] == pytest.approx(1.0)
    ledger = pilot.cost_ledger(contract, profile)
    assert ledger["llm_agent_arm_agent_steps"] == 3 * 16 * 256 * 18
    assert ledger["by_model"]["qwen3_0_6b"]["llm_agent_arm_gpu_hours"]["batch_1"] == pytest.approx(
        0.01 * 3 * 16 * 256 * 18 / 3600
    )


@needs_predecessors
def test_loader_rejects_an_elicitation_under_another_readout(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    root = _fake_root(tmp_path, contract)
    other = copy.deepcopy(contract)
    other["elicitation"]["readout"]["assistant_prefill"] = "Verdict: **"
    with pytest.raises(RuntimeError, match="different readout"):
        pilot.load_elicitations(other, root, "0" * 40)


def test_hand_task_nests_the_predecessor_without_exploration() -> None:
    contract = pilot.load_contract(CONFIG)
    rows = pilot.run_task(contract, "smoke", 0, "linear_fit", "hand", FAKE_PRIORS)
    assert [row["arm_id"] for row in rows] == [arm for arm, _ in pilot.ABM_ARMS]
    assert all(row["exploratory_entry_rate"] == 0.0 for row in rows)
    hybrid = pilot.run_task(contract, "smoke", 0, "axis_interaction_fit", "hybrid_qwen3_0_6b", FAKE_PRIORS)
    assert hybrid[0]["epsilon"] > 0.0


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = pilot.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    conditions = ("hand", "hybrid_qwen3_1_7b")
    serial = pilot.run_study(reduced, "smoke", FAKE_PRIORS, workers=1, conditions=conditions)
    parallel = pilot.run_study(reduced, "smoke", FAKE_PRIORS, workers=3, conditions=conditions)
    assert len(serial) == 3 * len(conditions) * len(pilot.ABM_ARMS)
    assert serial == parallel


def _rows(overrides: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in pilot.FIT_WORLDS:
            for condition in pilot.CONDITION_IDS:
                for arm, evidence in pilot.ABM_ARMS:
                    base = overrides.get(condition, {})
                    rows.append(
                        {
                            "replicate": replicate,
                            "world_id": world,
                            "condition_id": condition,
                            "arm_id": arm,
                            "evidence_regime": evidence,
                            "productive_share_of_entrants": base.get("share", 0.30) + wobble,
                            "probe_brier": base.get("brier", 0.10) + wobble,
                        }
                    )
    return rows


def test_decision_classifies_the_ladder_the_frontier_and_the_replication() -> None:
    contract = pilot.load_contract(CONFIG)
    overrides = {
        "hybrid_qwen3_4b_2507": {"share": 0.40, "brier": 0.101},
        "hybrid_v3_reference": {"share": 0.40, "brier": 0.101},
        "hybrid_qwen3_4b": {"share": 0.39, "brier": 0.102},
        "hybrid_qwen3_1_7b": {"share": 0.36, "brier": 0.103},
        "hybrid_qwen3_0_6b": {"share": 0.31, "brier": 0.09},
        "selective_qwen3_4b": {"share": 0.45, "brier": 0.13},
    }
    decision = pilot.decide(_rows(overrides), contract)
    by_model = decision["by_model"]
    assert by_model["qwen3_4b"]["hybrid_summary"] == "DOMINATES_IN_ALL_CELLS"
    assert by_model["qwen3_4b"]["anchor_summary"] == "MATCHES_ANCHOR_IN_ALL_CELLS"
    assert by_model["qwen3_1_7b"]["hybrid_summary"] == "DOMINATES_IN_ALL_CELLS"
    assert by_model["qwen3_1_7b"]["anchor_summary"] == "MATCHES_ANCHOR_NOWHERE"
    cell = by_model["qwen3_1_7b"]["anchor_cells"]["linear_fit/mlp_batch"]
    assert cell["disposition"] == "PRECISION_SHORTFALL"
    assert by_model["qwen3_0_6b"]["hybrid_summary"] == "DOMINATES_NOWHERE"
    assert by_model["qwen3_0_6b"]["hybrid_cells"]["linear_fit/mlp_batch"]["disposition"] == "LEARNING_ONLY"
    assert by_model["qwen3_4b"]["selective_summary"] == "SELECTS_BETTER_IN_ALL_CELLS"
    assert "anchor_summary" not in by_model["qwen3_4b_2507"]
    frontier = decision["size_frontier"]
    assert frontier["smallest_same_release_model_dominating_in_majority"] == "qwen3_1_7b"
    assert decision["v3_replication"] == "V3_HYBRID_20_MAJORITY_REPLICATED_ON_FRESH_SEEDS"


def test_development_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    template = pilot.run_task(contract, "smoke", 0, "linear_fit", "hand", FAKE_PRIORS)
    rows = [
        {**row, "replicate": replicate, "world_id": world, "condition_id": condition}
        for replicate in range(16)
        for world in pilot.FIT_WORLDS
        for condition in pilot.CONDITION_IDS
        for row in template
    ]
    entry = {
        "validity": {"median_answer_token_mass": 1.0},
        "spearman_readout_versus_anchor": 1.0,
        "calibrated_prior_design_cohort_auc": {world: 0.5 for world in pilot.FIT_WORLDS},
        "seconds_per_query": {"batch_1": 0.01, "batch_32": 0.002},
        "one_time_elicitation_gpu_seconds": 10.0,
    }
    phase_a = {"models": {source: dict(entry) for source in pilot.PRIOR_SOURCES}}
    summary = pilot.summarize(rows, contract, "development", FAKE_PRIORS, phase_a)
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["cells"]) == 3 * len(pilot.CONDITION_IDS) * len(pilot.ABM_ARMS)
    root = tmp_path / "elicitation"
    for model in pilot.MODEL_IDS:
        (root / model).mkdir(parents=True)
        for name in pilot.ELICITATION_FILES:
            (root / model / name).write_text("{}", encoding="utf-8")
    receipt = pilot.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
        elicitation_root=root,
    )
    assert receipt["row_count"] == 16 * 3 * len(pilot.CONDITION_IDS) * len(pilot.ABM_ARMS)
    assert set(receipt["elicitations"]) == set(pilot.MODEL_IDS)


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = pilot.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        pilot.delayed.verify_precommit("short", contract)
