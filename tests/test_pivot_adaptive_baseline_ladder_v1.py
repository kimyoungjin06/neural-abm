from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "configs" / "pivot_adaptive_baseline_ladder_v1.yaml"
SCRIPT = ROOT / "scripts" / "prototype_pivot_adaptive_baseline_ladder_v1.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("pivot_baseline_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract(module: object) -> dict[str, object]:
    return module._contract_with_transition(module.load_contract(CONFIG))


def test_contract_is_fair_threshold_free_and_nonselective() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert [row["id"] for row in config["model_ladder"]] == [
        "fixed_rule",
        "delta_imitative",
        "delta_cautionary",
        "neural_imitative",
        "neural_cautionary",
    ]
    assert config["delta_rule"]["eta"] == 0.5
    assert config["delta_rule"]["prior_anchor"] == 0.25
    assert config["delta_rule"]["feature_specific_weights"] is False
    assert config["delta_rule"]["backpropagation"] is False
    targets = config["common_target_vector"]
    assert targets["decision_thresholds"] == "NONE_DESCRIPTIVE_ONLY"
    assert targets["model_selection"] == "FORBIDDEN"
    assert config["boundaries"]["empirical_rows_read"] == 0
    assert config["boundaries"]["model_selected"] == "NO"


def test_fixed_and_neural_models_exactly_nest_persistence_predecessor() -> None:
    module = _module()
    contract = _contract(module)
    profile = {**contract["profiles"]["smoke"], "replicates": 1}
    config = module.study2.LearningStudyConfig(
        **profile,
        **contract["inherited_learning_parameters"],
        arms=("frozen", "imitative", "cautionary"),
    )
    scenario = module._scenario(contract)
    visibility = module._regimes(contract)[0]
    for model_id, mode in (
        ("fixed_rule", "frozen"),
        ("neural_imitative", "imitative"),
        ("neural_cautionary", "cautionary"),
    ):
        nested = module.run_model(
            model_id=model_id,
            scenario=scenario,
            config=config,
            visibility=visibility,
            contract=contract,
            replicate=0,
        )
        predecessor = module.persistence.run_persistence_arm(
            scenario=scenario,
            config=config,
            visibility=visibility,
            transition=contract["common_environment_transition"],
            replicate=0,
            learning_mode=mode,
        )
        for key in (*module.FINAL_METRICS, *module.DISTANCE_TRAJECTORIES):
            assert nested[key] == predecessor[key]


def test_delta_rule_is_scalar_deterministic_and_non_gradient() -> None:
    module = _module()
    contract = _contract(module)
    profile = {**contract["profiles"]["smoke"], "replicates": 1}
    config = module.study2.LearningStudyConfig(
        **profile,
        **contract["inherited_learning_parameters"],
        arms=("frozen", "imitative", "cautionary"),
    )
    kwargs = {
        "scenario": module._scenario(contract),
        "config": config,
        "visibility": module._regimes(contract)[0],
        "transition": contract["common_environment_transition"],
        "delta_rule": contract["delta_rule"],
        "replicate": 0,
        "learning_mode": "imitative",
    }
    first = module._run_delta_arm(**kwargs)
    second = module._run_delta_arm(**kwargs)
    assert first == second
    assert first["adaptive_state_count_per_agent"] == 1
    assert first["gradient_based"] is False
    assert first["evidence_batch_count"] > 0
    assert first["evidence_event_count"] >= first["evidence_batch_count"]


def test_smoke_scorecard_retains_all_cells_and_distances() -> None:
    module = _module()
    contract = _contract(module)
    rows, summary = module.run_model_ladder(contract, "smoke")
    assert len(rows) == 2 * 4 * 5
    assert len(summary["cells"]) == 4 * 5
    assert len(summary["delta_neural_distances"]) == 4 * 2
    assert summary["status"] == "DESCRIPTIVE_SYNTHETIC_MODEL_COMPARISON"
    assert summary["boundaries"]["model_selected"] == "NO"
    for report in summary["delta_neural_distances"]:
        assert report["final_vector_mae"]["n"] == 2
        assert report["trajectory_rmse"]["n"] == 2


def test_fixed_rule_is_visibility_invariant_in_scorecard() -> None:
    module = _module()
    contract = _contract(module)
    rows, _summary = module.run_model_ladder(contract, "smoke")
    fixed = [row for row in rows if row["model_id"] == "fixed_rule"]
    by_replicate: dict[int, set[tuple[float, ...]]] = {0: set(), 1: set()}
    for row in fixed:
        by_replicate[row["replicate"]].add(
            tuple(row[metric] for metric in module.FINAL_METRICS)
        )
    assert all(len(values) == 1 for values in by_replicate.values())


def test_source_bindings_and_exclusive_receipt(tmp_path: Path) -> None:
    module = _module()
    contract = _contract(module)
    verified = module.delayed.verify_source_bindings(contract)
    assert len(verified) == len(contract["source_bindings"])
    rows, summary = module.run_model_ladder(contract, "smoke")
    receipt = module.write_outputs(
        output_dir=tmp_path / "model-scorecard",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["row_count"] == 40
    persisted = json.loads(
        (tmp_path / "model-scorecard" / "receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == receipt
    with pytest.raises(FileExistsError):
        module.write_outputs(
            output_dir=tmp_path / "model-scorecard",
            rows=rows,
            summary=summary,
            precommit={"commit_sha": "0" * 40},
            contract_path=CONFIG,
        )
