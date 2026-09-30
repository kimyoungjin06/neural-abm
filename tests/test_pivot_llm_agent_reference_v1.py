from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_agent_reference_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_llm_agent_reference_v1 as n06  # noqa: E402


_RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
V4_SCORECARD = Path(_RAW["v4_prior"]["scorecard"])
needs_v4 = pytest.mark.skipif(not V4_SCORECARD.exists(), reason="v4 outputs are machine-local")
FAKE_PRIOR = {"weights": [0.5, 2.0, 1.5, -2.0, -1.0, 0.5, 0.0, 0.0, 0.0], "bias": -1.5}


class FakePolicy:
    """A deterministic stand-in for the LLM: log-odds from a hash of the prompt."""

    def __init__(self) -> None:
        self.cache: dict[str, float] = {}
        self.forward_passes = 0
        self.gpu_seconds = 0.0

    def _forward(self, prompt: str) -> tuple[float, float]:
        self.forward_passes += 1
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        return (digest / 2**32 - 0.5) * 4.0 - 1.0, 1.0

    def log_odds(self, prompts):  # noqa: ANN001, ANN201
        for prompt in prompts:
            if prompt not in self.cache:
                self.cache[prompt] = self._forward(prompt)[0]
        return np.asarray([self.cache[prompt] for prompt in prompts])

    def answer_mass(self, prompts):  # noqa: ANN001, ANN201
        return np.ones(len(prompts))


def _raw() -> dict:
    return copy.deepcopy(_RAW)


def _event(agent_id: int, features, productive: bool):  # noqa: ANN001, ANN202
    return SimpleNamespace(agent_id=agent_id, features=np.asarray(features, dtype=float), productive=productive)


def test_contract_plan_binds_sources_and_renders_the_agent_prompt() -> None:
    contract = n06.load_contract(CONFIG)
    plan = n06._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 14
    assert plan["conditions"] == ["hand", "hybrid_qwen3_4b_2507", "llm_agent_qwen3_4b_2507"]
    assert "No pivot outcomes in this field are public yet." in plan["example_prompt"]
    assert plan["example_prompt"].endswith("Answer with Yes or No.")


def test_contract_rejects_batching_reused_seeds_tuned_shares_and_paid_calls() -> None:
    contract = _raw()
    contract["runtime"]["batch_size"] = 32
    with pytest.raises(ValueError, match="batch"):
        n06.validate_contract(contract)
    contract = _raw()
    contract["probe_cohort"]["base_seed"] = 20261026
    with pytest.raises(ValueError, match="fresh seed"):
        n06.validate_contract(contract)
    contract = _raw()
    contract["conditions"][2]["exploratory_share"] = 0.0
    with pytest.raises(ValueError, match="exploratory share"):
        n06.validate_contract(contract)
    contract = _raw()
    contract["boundaries"]["paid_api_calls"] = 1
    with pytest.raises(ValueError, match="paid API"):
        n06.validate_contract(contract)


def test_evidence_text_counts_the_field_and_the_nearest_profiles_without_self() -> None:
    contract = n06.load_contract(CONFIG)
    assert n06.evidence_text(contract, [], np.zeros(9), 3) == contract["agent_prompt"]["no_evidence_text"]
    events = [_event(0, np.zeros(9), True), _event(1, np.full(9, 0.1), False)]
    events += [_event(10 + i, np.full(9, 0.9), False) for i in range(8)]
    text = n06.evidence_text(contract, events, np.zeros(9), exclude_agent_id=0)
    assert "0 researchers who pivoted established a productive line and 9 did not" in text
    assert "1 were productive" not in text
    own = n06.evidence_text(contract, events, np.zeros(9), exclude_agent_id=None)
    assert "1 researchers who pivoted established a productive line and 9 did not" in own
    assert "1 were productive and 7 were not" in own


def test_calibration_shift_hits_the_stated_base_rate() -> None:
    odds = np.linspace(-6.0, 1.0, 101)
    shift = n06.calibration_shift(odds, 0.30, [-40.0, 40.0], 1e-12)
    assert float(np.mean(1.0 / (1.0 + np.exp(-(odds + shift))))) == pytest.approx(0.30, abs=1e-10)
    with pytest.raises(RuntimeError, match="outside"):
        n06.calibration_shift(odds, 0.30, [5.0, 6.0], 1e-12)


def test_phase_a_calibrates_on_the_design_cohort_without_evidence() -> None:
    contract = n06.load_contract(CONFIG)
    result = n06.phase_a(contract, FakePolicy())
    assert result["profiles"] == 1024
    assert result["achieved_mean"] == pytest.approx(0.30, abs=1e-9)
    assert result["determinism"]["repeat_identical"] is True


def test_llm_agent_machinery_nests_the_fixed_rule_exactly() -> None:
    contract = n06.load_contract(CONFIG)
    n06.check_llm_machinery_nests_the_fixed_rule(contract, "smoke", 0, "linear_fit")
    assert n06.study2.PivotPolicyAgent is n06.ORIGINAL_AGENT


def test_llm_agent_run_is_deterministic_restores_slots_and_scores_the_probe() -> None:
    contract = n06.load_contract(CONFIG)
    first_policy, second_policy = FakePolicy(), FakePolicy()
    first = n06.run_llm_agent(contract, "smoke", 0, "axis_interaction_fit", n06.LLMField(contract, first_policy, 0.5))
    second = n06.run_llm_agent(contract, "smoke", 0, "axis_interaction_fit", n06.LLMField(contract, second_policy, 0.5))
    assert first == second
    assert n06.study2.PivotPolicyAgent is n06.ORIGINAL_AGENT
    assert n06.persistence.PersistenceCommitAdapter is n06.pilot3.ORIGINAL_ADAPTER
    assert first["agent_steps_queried"] >= 24
    probe = int(contract["probe_cohort"]["agent_count"])
    assert 0 < first_policy.forward_passes <= first["agent_steps_queried"] + probe
    hand = n06.abm2.run_arm(
        contract, n06.abm1.study_config(contract, "smoke"), replicate=0, visibility_id="immediate",
        world_id="axis_interaction_fit", arm_id="fixed_rule", evidence_regime="none",
        probe=n06.abm2.probe_cohort(contract, 0, "axis_interaction_fit"),
    )
    assert first["probe_brier"] != hand["probe_brier"]


def test_nabm_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = n06.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    serial, _ = n06.run_nabm(reduced, "smoke", FAKE_PRIOR, workers=1)
    parallel, _ = n06.run_nabm(reduced, "smoke", FAKE_PRIOR, workers=3)
    assert len(serial) == 3 * len(n06.NABM_CONDITIONS) * len(n06.NABM_ARMS)
    assert serial == parallel


def _rows(nabm: dict, llm: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in n06.FIT_WORLDS:
            for condition in n06.NABM_CONDITIONS:
                for arm, evidence in n06.NABM_ARMS:
                    base = nabm.get(condition, {})
                    rows.append(
                        {
                            "replicate": replicate, "world_id": world, "condition_id": condition,
                            "arm_id": arm, "evidence_regime": evidence,
                            "productive_share_of_entrants": base.get("share", 0.30) + wobble,
                            "probe_brier": base.get("brier", 0.10) + wobble,
                        }
                    )
            rows.append(
                {
                    "replicate": replicate, "world_id": world, "condition_id": n06.LLM_CONDITION,
                    "arm_id": n06.LLM_ARM[0], "evidence_regime": n06.LLM_ARM[1],
                    "productive_share_of_entrants": llm["share"] + wobble,
                    "probe_brier": llm["brier"] + wobble,
                }
            )
    return rows


def test_decision_classifies_nabm_against_the_llm_agent_and_the_llm_against_hand() -> None:
    contract = n06.load_contract(CONFIG)
    matched = n06.decide(
        _rows({"hybrid_qwen3_4b_2507": {"share": 0.39, "brier": 0.100}}, {"share": 0.40, "brier": 0.102}),
        contract,
    )
    assert matched["nabm_versus_llm_agent"]["summary"] == "NABM_MATCHES_LLM_AGENT_IN_ALL_CELLS"
    assert matched["llm_agent_versus_hand"]["summary"] == "DOMINATES_IN_ALL_CELLS"
    better = n06.decide(
        _rows({"hybrid_qwen3_4b_2507": {"share": 0.34, "brier": 0.100}}, {"share": 0.40, "brier": 0.08}),
        contract,
    )
    cell = better["nabm_versus_llm_agent"]["cells"]["linear_fit/mlp_batch"]
    assert cell["disposition"] == "LLM_AGENT_BETTER_BOTH"
    assert cell["nabm_precision_superior"]["pass"] is False


def test_summary_outputs_keep_timing_outside_the_receipt(tmp_path: Path) -> None:
    contract = n06.load_contract(CONFIG)
    rows = _rows({}, {"share": 0.35, "brier": 0.1})
    for row in rows:
        row.update(
            {"cumulative_entry_rate": 0.5, "exploratory_entry_rate": 0.1, "probe_auc": 0.6,
             "active_failed_rate": 0.1, "evidence_events_per_learner": 100.0, "agent_steps_queried": 3000}
        )
    summary = n06.summarize(rows, contract, "development", {"calibration_shift": 1.0})
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert summary["query_profile"]["mean_agent_steps_queried_per_run"] == 3000
    receipt = n06.write_outputs(
        output_dir=tmp_path / "out", rows=rows, summary=summary,
        precommit={"commit_sha": "0" * 40}, contract_path=CONFIG, timing={"wall": 1.0},
    )
    assert "cost_timing.json" not in receipt["outputs"]
    assert json.loads((tmp_path / "out" / "cost_timing.json").read_text())["wall"] == 1.0


@needs_v4
def test_cost_timing_uses_the_bound_v4_batch_32_speeds() -> None:
    contract = n06.load_contract(CONFIG)
    timing = n06.cost_timing(
        contract, "development", [4.0, 8.0],
        [{"wall_seconds": 10.0, "gpu_seconds": 9.0, "forward_passes": 1000}],
        None, [{"condition_id": n06.LLM_CONDITION}],
    )
    assert timing["nabm_cpu_seconds_per_arm_run"]["median"] == pytest.approx(1.5)
    floor = timing["cost_floor_from_v4_batch_32"]["qwen3_0_6b"]
    assert floor["gpu_seconds_per_run_at_this_query_count"] == pytest.approx(floor["seconds_per_query"] * 1000)
    assert n06.load_v4_prior(contract)["weights"]


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = n06.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        n06.delayed.verify_precommit("short", contract)
