from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments"
    / "configs"
    / "pivot_delayed_feedback_learning_prototype_v1.yaml"
)
SCRIPT = ROOT / "scripts" / "prototype_pivot_delayed_feedback_learning_v1.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("pivot_delayed_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_keeps_the_prototype_narrow_and_descriptive() -> None:
    config = _config()
    assert config["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert config["scenario"]["name"] == "hot_field_hype"
    assert config["learning_arms"] == ["frozen", "imitative", "cautionary"]
    assert [row["id"] for row in config["visibility_regimes"]] == [
        "immediate",
        "symmetric_lag_2",
        "failure_lag_2",
        "failure_lag_4",
        "selective_failure_lag_4",
    ]
    assert config["reporting"]["decision_thresholds"] == "NONE_DESCRIPTIVE_ONLY"
    assert config["reporting"]["peak_rule"] == (
        "NO_SINGLE_ARGMAX_OR_PEAK_CLASSIFICATION_IN_V1"
    )
    assert config["boundaries"] == {
        "repository_only": True,
        "generic_nabm_api_changed": False,
        "empirical_rows_read": 0,
        "empirical_fit": "NOT_RUN",
        "model_selection": "NOT_RUN",
        "neural_necessity": "NOT_ESTABLISHED",
        "scientific_effect": "NOT_ESTABLISHED",
        "package_promotion": "NOT_AUTHORIZED",
    }


def test_delay_zero_releases_on_the_next_step_and_delay_two_waits() -> None:
    module = _module()
    study2 = module.study2
    config = study2.LearningStudyConfig(
        agent_count=4,
        steps=5,
        burn_in_steps=1,
        replicates=1,
        neighbor_count=1,
    )
    event = study2.PivotEvent(
        agent_id=1,
        features=module.np.zeros(len(study2.FEATURES)),
        productive=False,
        step=1,
    )
    immediate = module.DelayedPivotEnvironment(
        neighbors=[[1], [0], [3], [2]],
        config=config,
        replicate=0,
        visibility=module.VisibilityRegime("immediate", 0, 0, 1.0, 1.0),
        step=1,
        new_events=[event],
    )
    immediate.advance()
    assert immediate.step == 2
    assert immediate.private_events == [event]
    assert immediate.last_events == [event]

    delayed = module.DelayedPivotEnvironment(
        neighbors=[[1], [0], [3], [2]],
        config=config,
        replicate=0,
        visibility=module.VisibilityRegime("lag", 0, 2, 1.0, 1.0),
        step=1,
        new_events=[event],
    )
    delayed.advance()
    assert delayed.last_events == []
    delayed.advance()
    assert delayed.last_events == []
    delayed.advance()
    assert delayed.step == 4
    assert delayed.last_events == [event]


def test_immediate_regime_exactly_nests_predecessor_core_outcomes() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    profile = {**contract["profiles"]["smoke"], "replicates": 1}
    config = module.study2.LearningStudyConfig(
        **profile,
        **contract["inherited_learning_parameters"],
        arms=tuple(contract["learning_arms"]),
    )
    scenario = module._scenario(contract)
    regime = module._regimes(contract)[0]
    for arm in contract["learning_arms"]:
        predecessor = module.study2.run_arm(
            scenario=scenario,
            config=config,
            replicate=0,
            learning_mode=arm,
        )
        delayed = module.run_delayed_arm(
            scenario=scenario,
            config=config,
            visibility=regime,
            replicate=0,
            learning_mode=arm,
        )
        for key in (
            "pivot_rate",
            "productive_pivot_rate",
            "failed_pivot_rate",
            "productive_share_of_pivots",
            "cumulative_failed_rate",
            "cumulative_productive_rate",
            "mean_attention_weight",
        ):
            assert delayed[key] == predecessor[key]


def test_smoke_phase_map_is_deterministic_and_retains_all_cells() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    rows_a, summary_a = module.run_phase_map(contract, "smoke")
    rows_b, summary_b = module.run_phase_map(contract, "smoke")
    assert rows_a == rows_b
    assert summary_a == summary_b
    assert len(rows_a) == 2 * 5 * 3
    assert len(summary_a["cells"]) == 5 * 3
    assert len(summary_a["paired_comparisons"]) == (4 * 3 * 4) + (5 * 2 * 4)
    assert set(summary_a["cells"][0]["mean_trajectories"]) == {
        "realized_success_per_step",
        "realized_failure_per_step",
        "public_success_per_step",
        "public_failure_per_step",
        "public_backlog_per_step",
        "cumulative_failed_rate",
        "cumulative_productive_rate",
        "mean_attention_weight",
    }
    assert summary_a["status"] == "DESCRIPTIVE_SYNTHETIC_RESULT"
    assert summary_a["boundaries"]["model_selected"] == "NO"


def test_source_bindings_and_exclusive_output_receipt(tmp_path: Path) -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    verified = module.verify_source_bindings(contract)
    assert len(verified) == len(contract["source_bindings"])
    rows, summary = module.run_phase_map(contract, "smoke")
    receipt = module.write_outputs(
        output_dir=tmp_path / "phase-map",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["row_count"] == 30
    assert {row["path"] for row in receipt["artifacts"]} == {
        "replicates.jsonl",
        "phase_map.json",
        "phase_map.md",
    }
    persisted = json.loads(
        (tmp_path / "phase-map" / "receipt.json").read_text(encoding="utf-8")
    )
    assert persisted == receipt
    with pytest.raises(FileExistsError):
        module.write_outputs(
            output_dir=tmp_path / "phase-map",
            rows=rows,
            summary=summary,
            precommit={"commit_sha": "0" * 40},
            contract_path=CONFIG,
        )
