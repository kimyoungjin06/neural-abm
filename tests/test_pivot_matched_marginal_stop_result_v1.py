from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_matched_marginal_discriminator_v1_20260928.summary.json"
)
SCRIPT = ROOT / "scripts" / "prototype_pivot_matched_marginal_discriminator_v1.py"


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _module() -> object:
    spec = importlib.util.spec_from_file_location("pivot_matched_marginal_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stop_is_bound_to_the_frozen_v1_precommit() -> None:
    result = _result()
    assert result["status"] == "INVALID_IMPLEMENTATION_STOP_NO_RESULT"
    precommit = result["precommit"]
    assert precommit["commit_sha"] == "b960fe0fc39e1c82390ee66c70856f4142f4532a"
    assert _sha256(ROOT / precommit["contract_path"]) == precommit["contract_sha256"]
    assert _sha256(ROOT / precommit["runner_path"]) == precommit["runner_sha256"]


def test_stop_wrote_nothing_and_evaluated_no_question() -> None:
    result = _result()
    execution = result["execution"]
    assert execution["contract_failure_action"] == "INVALID_IMPLEMENTATION_STOP"
    assert execution["output_dir_created"] is False
    assert execution["artifacts_written"] == []
    assert execution["metrics_written_printed_or_inspected"] is False
    assert set(result["registered_result_disposition"].values()) == {"NOT_EVALUATED"}
    assert result["boundaries"]["model_selected"] == "NO"
    assert result["boundaries"]["neural_necessity"] == "NOT_ESTABLISHED"


def test_recorded_scalar_drift_reproduces_from_frozen_v1_code() -> None:
    module = _module()
    contract = module.load_contract(module.DEFAULT_CONFIG)
    model = module._model_contract(contract, "scalar_delta")
    diagnosis = _result()["diagnosis"]
    maximum = 0.0
    for profile in ("smoke", "development"):
        settings = module._profile(contract, profile)
        observed: dict[str, list[int]] = {pair: [] for pair in module.PAIR_WORLDS}
        for replicate in range(settings["replicate_count"]):
            train = module.make_cohort(
                contract,
                role="training",
                replicate=replicate,
                agent_count=settings["training_agent_count"],
            )
            evaluation = module.make_cohort(
                contract,
                role="evaluation",
                replicate=replicate,
                agent_count=settings["evaluation_agent_count"],
            )
            for pair_id, (left, right) in module.PAIR_WORLDS.items():
                left_prediction = module._fit_scalar_delta(
                    train.features,
                    module.world_labels(train, left),
                    evaluation.features,
                    model,
                )
                right_prediction = module._fit_scalar_delta(
                    train.features,
                    module.world_labels(train, right),
                    evaluation.features,
                    model,
                )
                if not np.array_equal(left_prediction, right_prediction):
                    observed[pair_id].append(replicate)
                maximum = max(
                    maximum,
                    float(np.max(np.abs(left_prediction - right_prediction))),
                )
        assert observed == diagnosis[f"{profile}_profile"]["non_identical_replicates"]
    assert maximum == diagnosis["maximum_absolute_prediction_difference"]
