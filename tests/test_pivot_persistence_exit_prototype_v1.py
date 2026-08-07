from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "configs" / "pivot_persistence_exit_prototype_v1.yaml"
SCRIPT = ROOT / "scripts" / "prototype_pivot_persistence_exit_v1.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("pivot_persistence_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_is_nested_domain_owned_and_descriptive() -> None:
    config = _config()
    assert config["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert [row["id"] for row in config["state_models"]] == [
        "absorbing_entry",
        "persistence_exit_v1",
    ]
    transition = config["transition_contract"]
    assert transition["entry_rule"] == (
        "EXACT_PREDECESSOR_PIVOT_THRESHOLD_AND_COMMIT_ORDER"
    )
    assert transition["failed_minimum_active_steps"] == 2
    assert transition["failed_continuation_score"] == {
        "productive_fit_weight": 0.5,
        "current_policy_output_weight": 0.5,
        "exit_below": 0.4,
    }
    assert transition["reentry_after_exit"] is False
    assert config["reporting"]["decision_thresholds"] == "NONE_DESCRIPTIVE_ONLY"
    assert config["boundaries"]["generic_nabm_api_changed"] is False
    assert config["boundaries"]["empirical_rows_read"] == 0


def test_smoke_map_preserves_entry_exactly_and_is_deterministic() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    rows_a, summary_a = module.run_persistence_map(contract, "smoke")
    rows_b, summary_b = module.run_persistence_map(contract, "smoke")
    assert rows_a == rows_b
    assert summary_a == summary_b
    assert len(rows_a) == 2 * 4 * 3 * 2
    assert len(summary_a["cells"]) == 2 * 4 * 3
    assert summary_a["entry_invariance"] == {
        "status": "PASS",
        "paired_rows_checked": 24,
        "maximum_final_absolute_difference": 0.0,
        "maximum_trajectory_absolute_difference": 0.0,
    }
    grouped = {
        (row["state_model"], row["regime"], row["arm"], row["replicate"]): row
        for row in rows_a
    }
    for regime in [row["id"] for row in contract["visibility_regimes"]]:
        for arm in contract["learning_arms"]:
            for replicate in range(2):
                absorbing = grouped[("absorbing_entry", regime, arm, replicate)]
                persistence = grouped[
                    ("persistence_exit_v1", regime, arm, replicate)
                ]
                assert persistence["cumulative_entry_rate"] == absorbing[
                    "cumulative_entry_rate"
                ]
                assert persistence["cumulative_entry_rate_trajectory"] == (
                    absorbing["cumulative_entry_rate_trajectory"]
                )


def test_productive_states_do_not_exit_and_failed_exit_conserves_entries() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    profile = {**contract["profiles"]["smoke"], "replicates": 1}
    study_config = module.study2.LearningStudyConfig(
        **profile,
        **contract["inherited_learning_parameters"],
        arms=tuple(contract["learning_arms"]),
    )
    run = module.run_persistence_arm(
        scenario=module._scenario(contract),
        config=study_config,
        visibility=module._regimes(contract)[0],
        transition=contract["transition_contract"],
        replicate=0,
        learning_mode="cautionary",
    )
    assert run["active_productive_rate"] == run[
        "active_productive_rate_trajectory"
    ][-1]
    assert run["cumulative_entry_rate"] == pytest.approx(
        run["active_rate"] + run["exited_failed_rate"]
    )
    assert run["active_rate"] == pytest.approx(
        run["active_productive_rate"] + run["active_failed_rate"]
    )
    assert all(value < 0.4 for value in run["continuation_scores_at_exit"])


def test_frozen_persistence_is_invariant_to_visibility() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    rows, _summary = module.run_persistence_map(contract, "smoke")
    frozen = [
        row
        for row in rows
        if row["state_model"] == "persistence_exit_v1" and row["arm"] == "frozen"
    ]
    by_replicate: dict[int, set[tuple[float, ...]]] = {0: set(), 1: set()}
    for row in frozen:
        by_replicate[row["replicate"]].add(
            tuple(row[metric] for metric in module.FINAL_METRICS)
        )
    assert all(len(values) == 1 for values in by_replicate.values())


def test_source_bindings_and_exclusive_receipt(tmp_path: Path) -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    verified = module.delayed.verify_source_bindings(contract)
    assert len(verified) == len(contract["source_bindings"])
    rows, summary = module.run_persistence_map(contract, "smoke")
    receipt = module.write_outputs(
        output_dir=tmp_path / "persistence-map",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["row_count"] == 48
    assert receipt["entry_invariance"]["status"] == "PASS"
    persisted = json.loads(
        (tmp_path / "persistence-map" / "receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == receipt
    with pytest.raises(FileExistsError):
        module.write_outputs(
            output_dir=tmp_path / "persistence-map",
            rows=rows,
            summary=summary,
            precommit={"commit_sha": "0" * 40},
            contract_path=CONFIG,
        )
