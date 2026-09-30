from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_persona_agent_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_llm_persona_agent_v1 as n07  # noqa: E402


_RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


class FakePolicy:
    """Deterministic stand-in for the LLM: log-odds from a hash of the prompt."""

    def __init__(self, offset: float = -1.0) -> None:
        self.cache: dict[str, float] = {}
        self.forward_passes = 0
        self.gpu_seconds = 0.0
        self.offset = offset

    def _forward(self, prompt: str) -> tuple[float, float]:
        self.forward_passes += 1
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        return (digest / 2**32 - 0.5) * 6.0 + self.offset, 1.0

    def log_odds(self, prompts):  # noqa: ANN001, ANN201
        for prompt in prompts:
            if prompt not in self.cache:
                self.cache[prompt] = self._forward(prompt)[0]
        return np.asarray([self.cache[prompt] for prompt in prompts])

    def answer_mass(self, prompts):  # noqa: ANN001, ANN201
        return np.ones(len(prompts))


def _raw() -> dict:
    return copy.deepcopy(_RAW)


def test_contract_plan_binds_sources_and_renders_persona_prompts() -> None:
    contract = n07.load_contract(CONFIG)
    plan = n07._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 16
    assert plan["example_pivot_prompt"].startswith("You are a researcher")
    assert "1 of your 4 closest collaborators" in plan["example_pivot_prompt"]
    assert plan["example_pivot_prompt"].endswith("Do you pivot into this field now? Answer with Yes or No.")
    assert "2 steps ago" in plan["example_stay_prompt"]


def test_contract_rejects_calibration_sampling_batching_and_reused_seeds() -> None:
    contract = _raw()
    contract["persona"]["calibration"] = "SHIFT_TO_BASE_RATE"
    with pytest.raises(ValueError, match="not calibrated"):
        n07.validate_contract(contract)
    contract = _raw()
    contract["persona"]["decision_rule"] = "SAMPLED"
    with pytest.raises(ValueError, match="greedily"):
        n07.validate_contract(contract)
    contract = _raw()
    contract["runtime"]["batch_size"] = 8
    with pytest.raises(ValueError, match="batch"):
        n07.validate_contract(contract)
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261027
    with pytest.raises(ValueError, match="fresh seed"):
        n07.validate_contract(contract)


def test_decision_machinery_nests_the_fixed_rule_exactly() -> None:
    contract = n07.load_contract(CONFIG)
    n07.check_decision_machinery_nests_the_fixed_rule(contract, "smoke", 0, "linear_fit")
    assert n07.study2.PivotPolicyAgent is n07.n06.ORIGINAL_AGENT
    assert n07.persistence.PersistenceCommitAdapter is n07.pilot3.ORIGINAL_ADAPTER


def test_persona_run_decides_by_the_llm_and_is_deterministic() -> None:
    contract = n07.load_contract(CONFIG)
    first = n07.run_persona_agent(contract, "smoke", 0, "axis_interaction_fit", n07.PersonaField(contract, FakePolicy()))
    second = n07.run_persona_agent(contract, "smoke", 0, "axis_interaction_fit", n07.PersonaField(contract, FakePolicy()))
    assert first == second
    assert first["pivot_questions"] > 0
    entered_by_decision = first["cumulative_entry_rate"] - first["exploratory_entry_rate"]
    assert first["decided_entry_rate"] == pytest.approx(entered_by_decision)
    assert first["probe_brier"] is None and 0.0 <= first["probe_auc"] <= 1.0
    never = n07.run_persona_agent(
        contract, "smoke", 0, "axis_interaction_fit", n07.PersonaField(contract, FakePolicy(offset=-20.0))
    )
    assert never["pivot_yes"] == 0
    assert never["decided_entry_rate"] == pytest.approx(0.0)


def test_social_line_counts_pivoted_collaborators() -> None:
    contract = n07.load_contract(CONFIG)
    assert n07.social_text(contract, 4, 3) == (
        "Collaborators who have already pivoted into this field: 3 of your 4 closest collaborators."
    )


def test_phase_a_reports_validity_without_tuning() -> None:
    contract = n07.load_contract(CONFIG)
    result = n07.phase_a(contract, FakePolicy())
    assert result["profiles"] == 1024
    assert result["determinism"]["repeat_identical"] is True
    assert 0.0 <= result["zero_evidence_yes_share"] <= 1.0


def _rows(nabm: dict, persona: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in n07.FIT_WORLDS:
            for condition in n07.NABM_CONDITIONS:
                for arm, evidence in n07.NABM_ARMS:
                    base = nabm.get(condition, {})
                    rows.append(
                        {
                            "replicate": replicate, "world_id": world, "condition_id": condition,
                            "arm_id": arm, "evidence_regime": evidence,
                            "productive_share_of_entrants": base.get("share", 0.30) + wobble,
                            "probe_auc": base.get("auc", 0.90) + wobble,
                        }
                    )
            rows.append(
                {
                    "replicate": replicate, "world_id": world, "condition_id": n07.PERSONA_CONDITION,
                    "arm_id": n07.PERSONA_ARM[0], "evidence_regime": n07.PERSONA_ARM[1],
                    "productive_share_of_entrants": persona["share"] + wobble,
                    "probe_auc": persona["auc"] + wobble,
                    "decided_entry_rate": persona["decided"],
                }
            )
    return rows


def test_decision_classifies_nabm_against_the_persona_and_flags_self_sealing() -> None:
    contract = n07.load_contract(CONFIG)
    matched = n07.decide(
        _rows({"hybrid_qwen3_4b_2507": {"share": 0.40, "auc": 0.95}}, {"share": 0.36, "auc": 0.90, "decided": 0.4}),
        contract,
    )
    assert matched["nabm_versus_persona"]["summary"] == "NABM_MATCHES_PERSONA_AGENT_IN_ALL_CELLS"
    cell = matched["nabm_versus_persona"]["cells"]["linear_fit/mlp_batch"]
    assert cell["nabm_precision_superior"]["pass"] is True
    assert matched["persona_versus_hand_precision"]["summary"] == "SELECTS_BETTER_IN_ALL_CELLS"
    assert matched["self_sealing_by_world"]["linear_fit"]["self_sealing"] is False
    worse = n07.decide(
        _rows({"hybrid_qwen3_4b_2507": {"share": 0.35, "auc": 0.85}}, {"share": 0.40, "auc": 0.95, "decided": 0.01}),
        contract,
    )
    assert worse["nabm_versus_persona"]["cells"]["axis_interaction_fit/tree_batch"]["disposition"] == "PERSONA_AGENT_BETTER_BOTH"
    assert worse["self_sealing_by_world"]["axis_interaction_fit"]["self_sealing"] is True


def test_summary_and_outputs_keep_timing_outside_the_receipt(tmp_path: Path) -> None:
    contract = n07.load_contract(CONFIG)
    rows = _rows({}, {"share": 0.35, "auc": 0.8, "decided": 0.3})
    for row in rows:
        row.update(
            {"cumulative_entry_rate": 0.5, "exploratory_entry_rate": 0.1, "probe_brier": None,
             "active_failed_rate": 0.1, "exited_failed_rate": 0.1, "evidence_events_per_learner": 100.0}
        )
        if row["condition_id"] == n07.PERSONA_CONDITION:
            row.update({"agent_steps_queried": 2000, "pivot_questions": 900, "pivot_yes": 90, "stay_questions": 50, "stay_no": 10,
                        "probe_mean_willingness": 0.2})
    summary = n07.summarize(rows, contract, "development", {"zero_evidence_yes_share": 0.1})
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert summary["decision_totals"]["pivot_questions"] == 900 * 48
    receipt = n07.write_outputs(
        output_dir=tmp_path / "out", rows=rows, summary=summary,
        precommit={"commit_sha": "0" * 40}, contract_path=CONFIG, timing={"wall": 1.0},
    )
    assert "cost_timing.json" not in receipt["outputs"]


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = n07.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        n07.delayed.verify_precommit("short", contract)
