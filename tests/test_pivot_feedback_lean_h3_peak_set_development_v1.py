from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_peak_set_development_v1.yaml"
)
DEVELOPER = ROOT / "scripts/develop_pivot_feedback_lean_h3_peak_set_estimator_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_peak_set_development_v1.py"


def _module() -> dict[str, object]:
    return runpy.run_path(str(DEVELOPER))


def _curve_counts(
    opportunities: list[int],
    entry_rates: list[float],
    persistence_rates: list[float],
) -> tuple[list[int], list[int]]:
    entries = [round(total * rate) for total, rate in zip(opportunities, entry_rates)]
    persistent = [
        round(total * rate) for total, rate in zip(entries, persistence_rates)
    ]
    return entries, persistent


def _panel(
    *,
    challenge_id: str = "fixture_clear_interior",
    entry_rates: list[float] | None = None,
    persistence_rates: list[float] | None = None,
    missing_position: int | None = None,
) -> dict[str, object]:
    entry = entry_rates or [0.01, 0.02, 0.10, 0.02, 0.01]
    persistence = persistence_rates or [0.10, 0.60, 0.10, 0.05, 0.02]
    episodes: list[dict[str, object]] = []
    for field_id in range(4):
        for episode_id in range(8):
            opportunities = [20_000] * 5
            if missing_position is not None:
                opportunities[missing_position] = 0
            entries, persistent = _curve_counts(opportunities, entry, persistence)
            episodes.append(
                {
                    "field_id": field_id,
                    "episode_id": episode_id,
                    "opportunities": opportunities,
                    "entries": entries,
                    "persistent": persistent,
                }
            )
    return {
        "challenge_id": challenge_id,
        "replicate": 0,
        "grid": [-2, -1, 0, 1, 2],
        "episodes": episodes,
    }


def test_contract_is_development_only_and_preserves_all_empirical_boundaries() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == (
        "AUTHORIZED_DEVELOPMENT_ONLY_NO_ESTIMATOR_OR_SUPPORT_SELECTION"
    )
    assert config["source_diagnostic"]["role"] == (
        "OPENED_DISCOVERY_ONLY_NOT_REUSABLE_AS_BLIND_CONFIRMATION"
    )
    assert config["candidate"]["posterior_peak"] == {
        "earliest_argmax_within_draw": True,
        "peak_set_credible_mass": 0.90,
        "minimum_modal_peak_probability": 0.60,
        "maximum_peak_set_grid_span": 1,
        "require_contiguous_peak_set": True,
        "boundary_modal_peak_action": "INVALID",
        "any_registered_time_denominator_zero_action": "INVALID",
    }
    authorization = config["authorization"]
    for key in (
        "source_or_empirical_row_read",
        "current_g1_row_read",
        "model_selection_row_read",
        "holdout_or_private_g4_read",
        "empirical_parameter_calibration",
        "confirmatory_execution",
    ):
        assert authorization[key] == "FORBIDDEN"
    assert authorization["estimator_selected"] == "NO"
    assert authorization["support_rule_selected"] == "NO"
    assert authorization["scientific_effect"] == "NOT_ESTABLISHED"


def test_plan_has_no_estimation_diagnostic_or_selection() -> None:
    module = _module()
    result = module["plan"](CONFIG, source_binding_scope="none")
    assert result["status"] == (
        "PASS_PEAK_SET_DEVELOPMENT_PLAN_NO_ESTIMATION_NO_DIAGNOSTIC"
    )
    assert result["candidate"] == "jeffreys_peak_set_field_balanced_v1"
    assert result["estimator_selected"] == "NO"
    assert result["support_rule_selected"] == "NO"
    assert result["scientific_effect"] == "NOT_ESTABLISHED"


def test_estimation_scope_verifies_only_the_public_panel_binding() -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected = config["source_diagnostic"]["challenge_panels_sha256"]
    calls: list[str] = []

    def fake_sha256(path: Path) -> str:
        calls.append(path.name)
        assert path.name == "challenge_panels.jsonl"
        return expected

    load_config = module["load_config"]
    load_config.__globals__["_sha256"] = fake_sha256
    load_config(CONFIG, source_binding_scope="panels")
    assert calls == ["challenge_panels.jsonl"]


def test_clear_interior_peaks_are_valid_and_propagate_peak_uncertainty() -> None:
    module = _module()
    config = module["load_config"](CONFIG, source_binding_scope="none")
    result = module["_estimate_panel"](config, _panel(), 399)
    assert result["valid"] is True
    assert result["valid_episodes"] == 32
    assert result["point_estimate"] == 1.0
    assert result["passed"] is True
    assert result["two_sided_lower"] <= result["point_estimate"]
    assert result["two_sided_upper"] >= result["point_estimate"]
    assert result["invalid_reason_counts"] == {}


def test_plateau_peak_fails_closed_as_diffuse() -> None:
    module = _module()
    config = module["load_config"](CONFIG, source_binding_scope="none")
    panel = _panel(
        challenge_id="fixture_plateau",
        entry_rates=[0.01, 0.08, 0.08, 0.01, 0.005],
    )
    result = module["_estimate_panel"](config, panel, 399)
    assert result["valid"] is False
    assert result["passed"] is False
    assert result["point_estimate"] is None
    assert result["invalid_reason_counts"]["DIFFUSE_PEAK_PROBABILITY"] > 0
    assert result["invalid_reason_counts"][
        "INSUFFICIENT_VALID_EPISODES_PER_FIELD"
    ] == 1


def test_missing_registered_time_fails_closed() -> None:
    module = _module()
    config = module["load_config"](CONFIG, source_binding_scope="none")
    result = module["_estimate_panel"](
        config,
        _panel(challenge_id="fixture_missing", missing_position=2),
        199,
    )
    assert result["valid"] is False
    assert result["passed"] is False
    assert result["invalid_reason_counts"]["MISSING_REGISTERED_TIME"] == 32


def test_boundary_modal_peak_fails_closed() -> None:
    module = _module()
    config = module["load_config"](CONFIG, source_binding_scope="none")
    result = module["_estimate_panel"](
        config,
        _panel(
            challenge_id="fixture_boundary",
            entry_rates=[0.005, 0.01, 0.02, 0.04, 0.12],
        ),
        199,
    )
    assert result["valid"] is False
    assert result["passed"] is False
    assert result["invalid_reason_counts"]["BOUNDARY_MODAL_PEAK"] == 32


def test_panel_estimation_is_deterministic_for_same_identity() -> None:
    module = _module()
    config = module["load_config"](CONFIG, source_binding_scope="none")
    panel = _panel()
    first = module["_estimate_panel"](config, panel, 199)
    second = module["_estimate_panel"](config, copy.deepcopy(panel), 199)
    assert first == second
    encoded = repr(first).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == hashlib.sha256(encoded).hexdigest()


def test_development_tools_have_no_empirical_decode_surface() -> None:
    source = DEVELOPER.read_text(encoding="utf-8") + VERIFIER.read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "pyarrow",
        "duckdb",
        "pandas",
        "g1_risk_set_and_labels.parquet",
        "g4_author_cold_guard.parquet",
        ".iter_batches(",
    ):
        assert forbidden not in source
