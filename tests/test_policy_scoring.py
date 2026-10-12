from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from neural_abm.metrics import accuracy_from_probs, cross_entropy_from_probs
from neural_abm.policy_scoring import (
    POLICY_SCORING_METRICS,
    PolicyScoringLeakError,
    score_policies,
)


ROOT = Path(__file__).resolve().parents[1]


def _binary(seed: int = 0, n: int = 200, policies: int = 3) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.4).astype(int)
    predictions = {f"p{i}": np.clip(0.4 + 0.3 * (labels - 0.4) + 0.2 * rng.standard_normal(n), 0, 1) for i in range(policies)}
    return predictions, labels


def test_binary_scores_follow_the_registered_definitions() -> None:
    predictions, labels = _binary()
    report = score_policies(predictions, labels, metrics=POLICY_SCORING_METRICS, split="probe")
    assert report.split == "probe" and report.examples == 200 and report.classes == 2
    p0 = predictions["p0"]
    truth = np.where(labels == 1, p0, 1 - p0)
    scores = report.scores_of("p0")
    assert scores["brier"] == pytest.approx(np.mean((p0 - labels) ** 2))
    assert scores["log_loss"] == pytest.approx(np.mean(-np.log(truth + 1e-8)))
    assert scores["accuracy"] == pytest.approx(np.mean((p0 > 0.5) == labels))
    assert scores["auc"] == pytest.approx(roc_auc_score(labels, p0))
    ensemble = np.mean(list(predictions.values()), axis=0)
    assert report.ensemble_scores["auc"] == pytest.approx(roc_auc_score(labels, ensemble))
    assert report.mean_scores["brier"] == pytest.approx(np.mean([np.mean((p - labels) ** 2) for p in predictions.values()]))


def test_auc_handles_ties_like_sklearn_and_is_none_for_one_class() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    tied = np.array([0.2, 0.5, 0.5, 0.5, 0.9, 0.2, 0.1, 0.2])
    assert score_policies({"a": tied}, labels, metrics=("auc",)).scores_of("a")["auc"] == pytest.approx(roc_auc_score(labels, tied))
    report = score_policies({"a": tied}, np.zeros(8, dtype=int), metrics=("auc", "brier"))
    assert report.scores_of("a")["auc"] is None and report.mean_scores["auc"] is None
    assert report.mean_scores["brier"] is not None


def test_multiclass_brier_and_matrix_input() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.4, 0.3]])
    labels = np.array([0, 2, 0])
    scores = score_policies({"m": probs}, labels, metrics=("brier", "accuracy")).scores_of("m")
    onehot = np.eye(3)[labels]
    assert scores["brier"] == pytest.approx(np.mean(((probs - onehot) ** 2).sum(axis=1)))
    assert scores["accuracy"] == pytest.approx(2 / 3)
    binary_matrix = score_policies({"m": np.column_stack([1 - probs[:, 0], probs[:, 0]])}, [1, 0, 1], metrics=("brier",))
    binary_vector = score_policies({"m": probs[:, 0]}, [1, 0, 1], metrics=("brier",))
    assert binary_matrix.scores_of("m")["brier"] == pytest.approx(binary_vector.scores_of("m")["brier"])


def test_the_leak_check_refuses_a_policy_trained_on_a_held_out_unit() -> None:
    predictions, labels = _binary(n=5, policies=2)
    held_out = ["r10", "r11", "r12", "r13", "r14"]
    clean = score_policies(predictions, labels, held_out_ids=held_out, training_ids={"p0": ["r1", "r2"], "p1": []})
    assert clean.leak_checked is True
    assert score_policies(predictions, labels).leak_checked is False
    with pytest.raises(PolicyScoringLeakError, match="p1"):
        score_policies(predictions, labels, held_out_ids=held_out, training_ids={"p0": ["r1"], "p1": ["r12"]})
    with pytest.raises(ValueError, match="requires held_out_ids"):
        score_policies(predictions, labels, training_ids={"p0": [], "p1": []})
    with pytest.raises(ValueError, match="missing policies"):
        score_policies(predictions, labels, held_out_ids=held_out, training_ids={"p0": []})
    with pytest.raises(ValueError, match="distinct held-out units"):
        score_policies(predictions, labels, held_out_ids=held_out[:4] + ["r10"], training_ids={"p0": [], "p1": []})


@pytest.mark.parametrize(
    ("predictions", "labels", "kwargs", "message"),
    [
        ({}, [0], {}, "at least one policy"),
        ({"a": [0.2, 1.3]}, [0, 1], {}, r"\[0, 1\]"),
        ({"a": [0.2, 0.3]}, [0, 1, 1], {}, "must hold 3 examples"),
        ({"a": [0.2, 0.3]}, [0, 2], {}, r"labels must lie in \[0, 2\)"),
        ({"a": [0.2, 0.3]}, [0, 0.5], {}, "integer class indices"),
        ({"a": [0.2, 0.3]}, [0, 1], {"metrics": ("brier", "kappa")}, "metrics must be"),
        ({"a": [[0.2, 0.3, 0.5]]}, [0], {"metrics": ("auc",)}, "binary tasks only"),
        ({"a": [0.2], "b": [[0.2, 0.3, 0.5]]}, [0], {}, "same number of classes"),
    ],
)
def test_invalid_inputs_are_refused(predictions, labels, kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score_policies(predictions, labels, **kwargs)


def test_the_payload_is_json_ready() -> None:
    predictions, labels = _binary(n=20)
    payload = score_policies(predictions, labels, metrics=("brier", "auc")).to_payload()
    assert json.loads(json.dumps(payload)) == payload
    assert [p["policy_id"] for p in payload["policies"]] == ["p0", "p1", "p2"]


def test_toy1_consumer_parity() -> None:
    """Toy 1's previous torch metrics equal the scorer's accuracy and log loss."""

    generator = torch.Generator().manual_seed(3)
    probs = torch.softmax(torch.randn(64, 3, generator=generator), dim=-1)
    labels = torch.randint(0, 3, (64,), generator=generator)
    scores = score_policies({"agent": probs.numpy()}, labels.numpy(), metrics=("accuracy", "log_loss")).scores_of("agent")
    assert scores["accuracy"] == accuracy_from_probs(probs, labels)
    assert scores["log_loss"] == pytest.approx(cross_entropy_from_probs(probs, labels), rel=1e-6)


def test_pivot_abm_v2_consumer_parity() -> None:
    """Pivot ABM v2's probe Brier and AUC are the scorer's mean Brier and ensemble AUC."""

    script = ROOT / "scripts" / "prototype_pivot_nonlinear_policy_abm_v2.py"
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    abm2 = importlib.import_module(script.stem)
    predictions, labels = _binary(seed=7, n=300, policies=4)
    report = score_policies(predictions, labels, metrics=("brier", "auc"))
    probe = [predictions[k] for k in predictions]
    float_labels = labels.astype(np.float64)
    assert report.mean_scores["brier"] == pytest.approx(np.mean([abm2._brier(p, float_labels) for p in probe]))
    assert report.ensemble_scores["auc"] == pytest.approx(abm2._auc(np.mean(probe, axis=0), float_labels))


def test_the_module_imports_no_torch_scipy_or_sklearn() -> None:
    source = (ROOT / "src" / "neural_abm" / "policy_scoring.py").read_text(encoding="utf-8")
    for banned in ("torch", "scipy", "sklearn"):
        assert f"import {banned}" not in source and f"from {banned}" not in source
