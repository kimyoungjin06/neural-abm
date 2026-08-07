from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml"
)
DEVELOPER = ROOT / "scripts/develop_pivot_feedback_lean_h3_stochastic_set_v3.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_stochastic_set_v3.py"


def _module() -> dict[str, object]:
    return runpy.run_path(str(DEVELOPER))


def test_contract_is_public_disjoint_nonblind_and_preserves_all_boundaries() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["mode"] == (
        "PUBLIC_NONBLIND_DISJOINT_SYNTHETIC_DEVELOPMENT_NO_OPENED_CHALLENGE"
    )
    assert config["status"] == (
        "PRECOMMIT_REQUIRED_NO_GENERATION_NO_RESULT_NO_SELECTION"
    )
    assert config["scale_audit"]["criterion_fixed_without_opened_metrics"] is True
    assert config["development_reporting"]["thresholds"] == "NONE"
    assert config["dependence_boundary"]["repeated_author_dependence"] == (
        "UNRESOLVED_REQUIRES_EMPIRICAL_CLUSTER_IDENTITY_SCALE_AUDIT"
    )
    authorization = config["authorization"]
    for key in (
        "opened_challenge_panel_read",
        "opened_challenge_private_label_read",
        "source_or_empirical_row_read",
        "current_g1_row_read",
        "model_selection_row_read",
        "holdout_or_private_g4_read",
        "empirical_parameter_calibration",
        "repeated_author_model_claim",
        "confirmatory_execution",
    ):
        assert authorization[key] == "FORBIDDEN"
    assert authorization["candidate_admissibility"] == "NOT_EVALUATED"
    assert authorization["candidate_selected"] == "NO"
    assert authorization["estimator_selected"] == "NO"
    assert authorization["support_rule_selected"] == "NO"
    assert authorization["fresh_blind_authorized"] is False


def test_scale_audit_rejects_absolute_rule_and_retains_normalized_invariance() -> None:
    module = _module()
    config = module["load_config"](CONFIG)
    audit = module["_scale_audit"](config)
    assert audit == {
        "selection_criterion": "POSITIVE_AFFINE_RATE_TRANSFORMATION_INVARIANCE",
        "absolute_rule_invariant": False,
        "normalized_rule_invariant": True,
        "retained_rule": "normalized_prominence_regret",
        "status": "PASS_ANALYTIC_SCALE_INVARIANCE_AUDIT",
    }
    peak_set = module["_normalized_peak_set"]
    base = [0.01, 0.02, 0.10, 0.095, 0.01]
    transformed = [4.0 * value for value in base]
    first = peak_set(base, regret_margin=0.10, flat_floor=0.01)
    second = peak_set(transformed, regret_margin=0.10, flat_floor=0.01)
    assert first["peak_set"] == second["peak_set"] == [0.0, 1.0]


def test_truth_geometry_preserves_adjacent_and_multimodal_sets() -> None:
    module = _module()
    gap_truth = module["_episode_gap_truth"]
    persistence = [0.10, 0.45, 0.15, 0.10, 0.08]
    adjacent = gap_truth(
        [0.005, 0.015, 0.055, 0.050, 0.005],
        persistence,
        regret_margin=0.10,
        flat_floor=0.01,
    )
    assert adjacent["valid"] is True
    assert adjacent["interval"] == [1.0, 2.0]
    multimodal = gap_truth(
        [0.005, 0.055, 0.010, 0.0525, 0.005],
        persistence,
        regret_margin=0.10,
        flat_floor=0.01,
    )
    assert multimodal["valid"] is True
    assert multimodal["interval"] == [0.0, 2.0]
    assert multimodal["multimodal"] is True


def test_truth_geometry_keeps_flat_boundary_and_missing_invalid() -> None:
    peak_set = _module()["_normalized_peak_set"]
    flat = peak_set([0.050, 0.052, 0.055, 0.051, 0.050], regret_margin=0.10, flat_floor=0.01)
    assert flat == {"valid": False, "reason": "PRACTICALLY_FLAT_CURVE"}
    boundary = peak_set([0.005, 0.010, 0.015, 0.025, 0.055], regret_margin=0.10, flat_floor=0.01)
    assert boundary == {"valid": False, "reason": "BOUNDARY_PRACTICAL_PEAK_SET"}
    missing = peak_set([0.005, 0.015, None, 0.015, 0.005], regret_margin=0.10, flat_floor=0.01)
    assert missing == {"valid": False, "reason": "MISSING_REGISTERED_TIME"}


def test_field_aggregation_preserves_mixed_vector_without_binary_decision() -> None:
    module = _module()
    config = module["load_config"](CONFIG)
    episode_rows: list[dict[str, object]] = []
    for field_id, value in enumerate((-1.0, 0.0, 1.0, 2.0)):
        for episode_id in range(8):
            episode_rows.append(
                {
                    "field_id": field_id,
                    "episode_id": episode_id,
                    "valid": True,
                    "reasons": [],
                    "interval": [value, value],
                    "multimodal": False,
                }
            )
    result = module["_aggregate_episode_results"](
        config,
        "fixture_mixed",
        "fixture",
        0,
        episode_rows,
    )
    assert result["field_intervals"] == [
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    assert result["field_balanced_interval_hull"] == [0.5, 0.5]
    assert result["direction_state"] == (
        "MIXED_OR_PARTIALLY_IDENTIFIED_NO_BINARY_DECISION"
    )
    assert "passed" not in result


def test_posterior_peak_set_is_deterministic_and_missing_fails_closed() -> None:
    module = _module()
    posterior = module["_posterior_peak_set"]
    kwargs = {
        "successes": [100, 300, 1100, 300, 100],
        "trials": [20_000] * 5,
        "draws": 399,
        "regret_margin": 0.10,
        "inclusion_probability": 0.90,
        "flat_floor": 0.01,
        "flat_probability": 0.90,
    }
    first = posterior(**kwargs, rng=np.random.default_rng(17))
    second = posterior(**copy.deepcopy(kwargs), rng=np.random.default_rng(17))
    assert first == second
    assert first["valid"] is True
    assert first["peak_set"] == [0.0]
    missing = posterior(
        **{**kwargs, "trials": [20_000, 20_000, 0, 20_000, 20_000]},
        rng=np.random.default_rng(17),
    )
    assert missing == {"valid": False, "reason": "MISSING_REGISTERED_DENOMINATOR"}


def test_bad_or_uncommitted_precommit_is_rejected() -> None:
    module = _module()
    error = module["StochasticSetDevelopmentError"]
    with pytest.raises(error, match="full SHA-1"):
        module["_verify_precommit"]("not-a-commit")


def test_estimation_never_requires_or_reads_public_truth(tmp_path: Path) -> None:
    module = _module()
    episodes = []
    for field_id in range(4):
        for episode_id in range(8):
            episodes.append(
                {
                    "field_id": field_id,
                    "episode_id": episode_id,
                    "opportunities": [20_000] * 5,
                    "entries": [100, 300, 1_100, 300, 100],
                    "persistent": [10, 135, 165, 30, 8],
                }
            )
    panels = tmp_path / "public_panels.jsonl"
    panels.write_text(
        json.dumps(
            {
                "scenario_id": "no_truth_fixture",
                "group": "fixture",
                "replicate": 0,
                "grid": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "episodes": episodes,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generation = {
        "status": "PASS_PUBLIC_NONBLIND_DISJOINT_GENERATION",
        "profile": "smoke",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "producer_sha256": hashlib.sha256(DEVELOPER.read_bytes()).hexdigest(),
        "panels": {"sha256": hashlib.sha256(panels.read_bytes()).hexdigest()},
    }
    (tmp_path / "generation_receipt.json").write_text(
        json.dumps(generation),
        encoding="utf-8",
    )

    result = module["estimate"](CONFIG, tmp_path)

    assert not (tmp_path / "public_truth.jsonl").exists()
    assert result["status"] == "PASS_PUBLIC_PANEL_ESTIMATION_NO_TRUTH_READ"
    assert result["truth_files_read"] == 0


def test_tools_expose_no_opened_or_empirical_decode_surface() -> None:
    source = DEVELOPER.read_text(encoding="utf-8") + VERIFIER.read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "/ssd/",
        "challenge_panels.jsonl",
        "private_labels.json",
        "g1_risk_set_and_labels.parquet",
        "g4_author_cold_guard.parquet",
        "pyarrow",
        "duckdb",
        "pandas",
    ):
        assert forbidden not in source
