from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v3.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_llm_prior_pilot_v3 as pilot  # noqa: E402


PRIOR_PATH = Path(
    yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["v2_calibrated_priors"]["path"]
)
needs_v2 = pytest.mark.skipif(not PRIOR_PATH.exists(), reason="v2 priors are machine-local")
FAKE_PRIOR = {"weights": [0.5, 2.0, 1.5, -2.0, -1.0, 0.5, 0.0, 0.0, 0.0], "bias": -1.5}


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_plan_binds_sources_and_derives_epsilons() -> None:
    contract = pilot.load_contract(CONFIG)
    plan = pilot._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 12
    assert plan["epsilon_by_condition"]["hand"] == 0.0
    for condition in contract["conditions"]:
        epsilon = plan["epsilon_by_condition"][condition["id"]]
        assert 1.0 - (1.0 - epsilon) ** 16 == pytest.approx(condition["exploratory_share"])


def test_reused_seed_and_llm_calls_are_rejected() -> None:
    contract = _raw()
    contract["probe_cohort"]["base_seed"] = 20261021
    with pytest.raises(ValueError, match="fresh seed"):
        pilot.validate_contract(contract)
    contract = _raw()
    contract["boundaries"]["new_llm_calls"] = 1
    with pytest.raises(ValueError, match="no LLM calls"):
        pilot.validate_contract(contract)


@needs_v2
def test_llm_prior_is_the_hash_bound_v2_base_rate_prior(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    prior = pilot.load_llm_prior(contract)
    stored = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))["priors"]["llm_base_rate"]
    assert prior == {"weights": stored["weights"], "bias": stored["bias"]}
    copy_path = tmp_path / "calibrated_priors.json"
    shutil.copy(PRIOR_PATH, copy_path)
    copy_path.write_text(copy_path.read_text(encoding="utf-8").replace("1", "2", 1), encoding="utf-8")
    tampered = copy.deepcopy(contract)
    tampered["v2_calibrated_priors"]["path"] = str(copy_path)
    with pytest.raises(RuntimeError, match="changed"):
        pilot.load_llm_prior(tampered)


def test_slots_are_restored_even_on_error() -> None:
    adapter = pilot.exploration_adapter(0.5, 1, 0)
    with pytest.raises(ZeroDivisionError), pilot.pilot1.prior_slot(FAKE_PRIOR), pilot.adapter_slot(adapter):
        assert pilot.persistence.PersistenceCommitAdapter is adapter
        _ = 1 / 0
    assert pilot.persistence.PersistenceCommitAdapter is pilot.ORIGINAL_ADAPTER
    assert pilot.study2._initial_policy is pilot.pilot1.HAND_INITIAL_POLICY


def test_hand_task_nests_the_predecessor_without_exploration() -> None:
    contract = pilot.load_contract(CONFIG)
    rows = pilot.run_task(contract, "smoke", 0, "linear_fit", "hand", FAKE_PRIOR)
    assert [row["arm_id"] for row in rows] == [arm for arm, _ in pilot.ABM_ARMS]
    assert all(row["exploratory_entry_rate"] == 0.0 for row in rows)


def test_exploration_adds_prior_independent_entrants_deterministically() -> None:
    contract = copy.deepcopy(pilot.load_contract(CONFIG))
    for condition in contract["conditions"]:
        if condition["id"] == "hybrid_40":
            condition["exploratory_share"] = 0.9
    selective = pilot.run_task(contract, "smoke", 0, "axis_interaction_fit", "llm_selective", FAKE_PRIOR)
    exploring = pilot.run_task(contract, "smoke", 0, "axis_interaction_fit", "hybrid_40", FAKE_PRIOR)
    again = pilot.run_task(contract, "smoke", 0, "axis_interaction_fit", "hybrid_40", FAKE_PRIOR)
    assert exploring == again
    assert exploring[0]["exploratory_entry_rate"] > 0.0
    assert exploring[0]["cumulative_entry_rate"] > selective[0]["cumulative_entry_rate"]


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = pilot.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    serial = pilot.run_study(reduced, "smoke", FAKE_PRIOR, workers=1)
    parallel = pilot.run_study(reduced, "smoke", FAKE_PRIOR, workers=3)
    assert len(serial) == 3 * len(pilot.CONDITION_IDS) * len(pilot.ABM_ARMS)
    assert serial == parallel


def _rows(overrides: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in pilot.FIT_WORLDS:
            for condition in pilot.CONDITION_IDS:
                for arm, evidence in pilot.ABM_ARMS:
                    base = overrides.get((condition, arm), {})
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


def test_decision_classifies_dominance_and_restoration() -> None:
    contract = pilot.load_contract(CONFIG)
    overrides = {}
    for arm in pilot.FIELD_ARMS:
        overrides[("llm_selective", arm)] = {"share": 0.45, "brier": 0.13}
        overrides[("hybrid_20", arm)] = {"share": 0.40, "brier": 0.102}
        overrides[("hybrid_40", arm)] = {"share": 0.31, "brier": 0.09}
    overrides[("hybrid_10", "mlp_batch")] = {"share": 0.42, "brier": 0.10}
    decision = pilot.decide(_rows(overrides), contract)["by_condition"]
    assert decision["llm_selective"]["summary"] == "DOMINATES_NOWHERE"
    assert decision["llm_selective"]["cells"]["linear_fit/mlp_batch"]["disposition"] == "SELECTION_ONLY"
    assert decision["hybrid_20"]["summary"] == "DOMINATES_IN_ALL_CELLS"
    restored = decision["hybrid_20"]["cells"]["axis_interaction_fit/tree_batch"][
        "learning_restored_versus_selective"
    ]
    assert restored["pass"] is True
    assert decision["hybrid_40"]["cells"]["linear_fit/linear_batch"]["disposition"] == "LEARNING_ONLY"
    assert decision["hybrid_10"]["summary"] == "DOMINATES_IN_MINORITY"
    assert len(decision["hybrid_10"]["dominating_cells"]) == 3


def test_development_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    template = pilot.run_task(contract, "smoke", 0, "linear_fit", "hand", FAKE_PRIOR)
    rows = [
        {**row, "replicate": replicate, "world_id": world, "condition_id": condition}
        for replicate in range(16)
        for world in pilot.FIT_WORLDS
        for condition in pilot.CONDITION_IDS
        for row in template
    ]
    summary = pilot.summarize(rows, contract, "development")
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["cells"]) == 3 * len(pilot.CONDITION_IDS) * len(pilot.ABM_ARMS)
    receipt = pilot.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["row_count"] == 16 * 3 * len(pilot.CONDITION_IDS) * len(pilot.ABM_ARMS)


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = pilot.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        pilot.delayed.verify_precommit("short", contract)
