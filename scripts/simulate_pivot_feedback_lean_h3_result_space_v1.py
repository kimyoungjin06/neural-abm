#!/usr/bin/env python3
"""Run the source-independent lean-H3 prospective result-space simulator."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_result_space_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_lean_h3_result_space.v1"
REPORT_SCHEMA = "pivot_feedback_lean_h3_result_space_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_lean_h3_result_space_receipt.v1"
EXPECTED_MODE = "SYNTHETIC_ONLY_PROSPECTIVE_ESTIMATOR_AND_INTERPRETATION_STRESS_TEST"
EXPECTED_STATUS = (
    "AUTHORIZED_SYNTHETIC_ONLY_NO_EMPIRICAL_ROW_ACCESS_NO_SCIENTIFIC_CLAIM"
)
EXPECTED_EVENT_GRID = (-2, -1, 0, 1, 2)
EXPECTED_ESTIMATORS = (
    "raw_earliest_argmax",
    "kernel_earliest_argmax",
)
MECHANISM_FAMILIES = (
    "delayed_feedback",
    "synchronous_opportunity",
    "fast_correction",
    "selective_composition",
    "capacity_depletion",
    "recruitment_mixture",
    "measurement_artifact",
)
CURVE_SHAPES = ("sharp", "broad", "flat", "bimodal")
FLAG_WEIGHTS = {
    "directional_false_positive": 4,
    "delayed_feedback_false_negative": 1,
    "sign_reversal": 4,
    "confident_wrong_interval": 5,
    "nonidentifiable_truth_pass": 3,
    "mechanism_overreach_risk": 1,
    "aggregation_paradox": 2,
    "estimator_disagreement": 2,
}


class ResultSpaceError(RuntimeError):
    """Raised when a result-space contract or execution fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultSpaceError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_create_only(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"mode drift: {path}")


def _write_json(path: Path, payload: Any) -> None:
    _write_create_only(path, _json_bytes(payload))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )
    _write_create_only(path, data)


def _load_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "config root must be a mapping")
    return payload


def _finite_number(value: Any, label: str) -> float:
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def _positive_int(value: Any, label: str) -> int:
    require(
        not isinstance(value, bool) and isinstance(value, int) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def _range_pair(value: Any, label: str) -> tuple[float, float]:
    require(isinstance(value, list) and len(value) == 2, f"{label} must have two values")
    lower = _finite_number(value[0], f"{label}[0]")
    upper = _finite_number(value[1], f"{label}[1]")
    require(lower <= upper, f"{label} must be ordered")
    return lower, upper


def _normalize_scenario(
    raw: Mapping[str, Any],
    *,
    scenario_index: int,
    scenario_kind: str,
    field_count: int,
) -> dict[str, Any]:
    scenario = dict(raw)
    scenario_id = scenario.get("scenario_id")
    require(isinstance(scenario_id, str) and scenario_id, "scenario_id is required")
    mechanism = scenario.get("mechanism_family")
    require(mechanism in MECHANISM_FAMILIES, f"invalid mechanism: {mechanism}")
    require(isinstance(scenario.get("feedback_present"), bool), "feedback_present must be bool")
    require(isinstance(scenario.get("truth_identifiable"), bool), "truth_identifiable must be bool")

    defaults: dict[str, Any] = {
        "observed_entry_shift": 0.0,
        "observed_persistence_shift": 0.0,
        "field_entry_peak_offsets": [0.0] * field_count,
        "field_persistence_peak_offsets": [0.0] * field_count,
        "secondary_peak_offset": 2.0,
        "secondary_peak_weight": 0.75,
        "entry_width": None,
        "persistence_width": None,
        "entry_label_sensitivity": 1.0,
        "entry_label_specificity": 1.0,
        "persistence_label_sensitivity": 1.0,
        "persistence_label_specificity": 1.0,
        "persistence_censoring": 0.0,
    }
    for key, value in defaults.items():
        scenario.setdefault(key, value)

    for key in (
        "entry_peak",
        "persistence_peak",
        "observed_entry_shift",
        "observed_persistence_shift",
        "opportunity_level",
        "base_entry_probability",
        "entry_amplitude",
        "base_persistence_probability",
        "persistence_amplitude",
        "episode_peak_sd",
        "secondary_peak_offset",
        "secondary_peak_weight",
        "entry_label_sensitivity",
        "entry_label_specificity",
        "persistence_label_sensitivity",
        "persistence_label_specificity",
        "persistence_censoring",
    ):
        scenario[key] = _finite_number(scenario.get(key), f"{scenario_id}.{key}")

    for key in ("entry_shape", "persistence_shape"):
        require(scenario.get(key) in CURVE_SHAPES, f"invalid {scenario_id}.{key}")

    shape_widths = {"sharp": 0.55, "broad": 1.25, "flat": 1.0, "bimodal": 0.65}
    for prefix in ("entry", "persistence"):
        width_key = f"{prefix}_width"
        if scenario[width_key] is None:
            scenario[width_key] = shape_widths[str(scenario[f"{prefix}_shape"])]
        else:
            scenario[width_key] = _finite_number(
                scenario[width_key], f"{scenario_id}.{width_key}"
            )
        require(scenario[width_key] > 0.0, f"{scenario_id}.{width_key} must be positive")

    for key in ("field_entry_peak_offsets", "field_persistence_peak_offsets"):
        offsets = scenario[key]
        require(
            isinstance(offsets, list) and len(offsets) == field_count,
            f"{scenario_id}.{key} must match field_count",
        )
        scenario[key] = [
            _finite_number(value, f"{scenario_id}.{key}") for value in offsets
        ]

    require(scenario["opportunity_level"] >= 1, "opportunity_level must be >= 1")
    for key in (
        "base_entry_probability",
        "entry_amplitude",
        "base_persistence_probability",
        "persistence_amplitude",
        "entry_label_sensitivity",
        "entry_label_specificity",
        "persistence_label_sensitivity",
        "persistence_label_specificity",
        "persistence_censoring",
        "secondary_peak_weight",
    ):
        require(0.0 <= scenario[key] <= 1.0, f"{scenario_id}.{key} outside [0,1]")
    require(scenario["episode_peak_sd"] >= 0.0, "episode_peak_sd must be nonnegative")
    scenario["scenario_index"] = scenario_index
    scenario["scenario_kind"] = scenario_kind
    scenario["scenario_seed_code"] = (
        1_000 + scenario_index if scenario_kind == "canonical" else 100_000 + scenario_index
    )
    return scenario


def load_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("mode") == EXPECTED_MODE, "config mode drift")
    require(config.get("status") == EXPECTED_STATUS, "config status drift")
    authorization = config.get("authorization")
    require(isinstance(authorization, dict), "authorization missing")
    for key in (
        "source_or_empirical_row_read",
        "current_g1_row_read",
        "model_selection_row_read",
        "holdout_or_private_g4_read",
        "empirical_parameter_calibration",
        "confirmatory_execution",
    ):
        require(authorization.get(key) == "FORBIDDEN", f"authorization drift: {key}")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")

    scientific = config.get("scientific_object")
    require(isinstance(scientific, dict), "scientific_object missing")
    require(
        tuple(scientific.get("event_time_grid", ())) == EXPECTED_EVENT_GRID,
        "event-time grid drift",
    )
    design = config.get("design")
    require(isinstance(design, dict), "design missing")
    field_count = _positive_int(design.get("field_count"), "field_count")
    episodes = _positive_int(design.get("episodes_per_field"), "episodes_per_field")
    minimum_episodes = _positive_int(
        design.get("minimum_supported_episodes_per_field"),
        "minimum_supported_episodes_per_field",
    )
    require(minimum_episodes <= episodes, "minimum supported episodes exceed total")
    _positive_int(
        design.get("minimum_finite_event_times_per_curve"),
        "minimum_finite_event_times_per_curve",
    )
    require(
        tuple(design.get("estimators", ())) == EXPECTED_ESTIMATORS,
        "estimator surface drift",
    )
    weights = design.get("kernel_weights")
    require(isinstance(weights, list) and len(weights) == 3, "kernel weights drift")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    require(np.all(normalized_weights >= 0.0), "kernel weights must be nonnegative")
    require(np.isclose(normalized_weights.sum(), 1.0), "kernel weights must sum to one")
    profiles = design.get("profiles")
    require(isinstance(profiles, dict) and profiles, "profiles missing")
    for name, profile in profiles.items():
        require(isinstance(name, str) and isinstance(profile, dict), "invalid profile")
        for key in (
            "canonical_replicates",
            "adversarial_scenarios",
            "adversarial_replicates",
            "bootstrap_replicates",
        ):
            _positive_int(profile.get(key), f"profiles.{name}.{key}")

    require(tuple(config.get("mechanism_families", ())) == MECHANISM_FAMILIES, "mechanism surface drift")
    canonical = config.get("canonical_scenarios")
    require(isinstance(canonical, list) and canonical, "canonical scenarios missing")
    normalized = [
        _normalize_scenario(
            raw,
            scenario_index=index,
            scenario_kind="canonical",
            field_count=field_count,
        )
        for index, raw in enumerate(canonical)
    ]
    ids = [item["scenario_id"] for item in normalized]
    require(len(ids) == len(set(ids)), "duplicate canonical scenario_id")
    config["canonical_scenarios"] = normalized
    return config


def _uniform(rng: np.random.Generator, value: Sequence[Any]) -> float:
    lower, upper = _range_pair(list(value), "adversarial range")
    return float(rng.uniform(lower, upper))


def _random_scenarios(config: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    design = config["design"]
    search = config["adversarial_search"]
    field_count = int(design["field_count"])
    rng = np.random.default_rng(np.random.SeedSequence([int(design["base_seed"]), 91_001]))
    offset_patterns = search["field_offset_patterns"]
    shapes = tuple(search["shape_levels"])
    scenarios: list[dict[str, Any]] = []
    for index in range(count):
        family = MECHANISM_FAMILIES[index % len(MECHANISM_FAMILIES)]
        persistence_peak = float(rng.choice((-1, 0, 1)))
        entry_peak = persistence_peak
        feedback = family == "delayed_feedback"
        observed_entry_shift = 0.0
        observed_persistence_shift = 0.0
        entry_shape = str(rng.choice(shapes))
        persistence_shape = str(rng.choice(shapes))
        if family == "delayed_feedback":
            entry_peak = persistence_peak + float(rng.choice((1, 2)))
        elif family == "fast_correction":
            entry_peak = persistence_peak - float(rng.choice((1, 2)))
        elif family in {"selective_composition", "capacity_depletion"}:
            entry_peak = persistence_peak + float(rng.choice((1, 2)))
        elif family == "recruitment_mixture":
            entry_shape = "bimodal"
            entry_peak = persistence_peak
        elif family == "measurement_artifact":
            observed_entry_shift = float(rng.choice((-1, 1)))
        truth_identifiable = not (
            entry_shape == "flat" or persistence_shape == "flat"
        )
        raw = {
            "scenario_id": f"adversarial_{index:04d}_{family}",
            "mechanism_family": family,
            "feedback_present": feedback,
            "truth_identifiable": truth_identifiable,
            "entry_peak": entry_peak,
            "persistence_peak": persistence_peak,
            "observed_entry_shift": observed_entry_shift,
            "observed_persistence_shift": observed_persistence_shift,
            "entry_shape": entry_shape,
            "persistence_shape": persistence_shape,
            "entry_width": _uniform(rng, search["peak_width_range"]),
            "persistence_width": _uniform(rng, search["peak_width_range"]),
            "secondary_peak_offset": float(rng.choice((-2, 2))),
            "secondary_peak_weight": float(rng.uniform(0.50, 1.00)),
            "opportunity_level": int(
                round(
                    math.exp(
                        rng.uniform(
                            math.log(search["opportunity_level_range"][0]),
                            math.log(search["opportunity_level_range"][1]),
                        )
                    )
                )
            ),
            "base_entry_probability": _uniform(
                rng, search["base_entry_probability_range"]
            ),
            "entry_amplitude": _uniform(rng, search["entry_amplitude_range"]),
            "base_persistence_probability": _uniform(
                rng, search["base_persistence_probability_range"]
            ),
            "persistence_amplitude": _uniform(
                rng, search["persistence_amplitude_range"]
            ),
            "episode_peak_sd": _uniform(rng, search["episode_peak_sd_range"]),
            "field_entry_peak_offsets": list(
                offset_patterns[int(rng.integers(0, len(offset_patterns)))]
            ),
            "field_persistence_peak_offsets": [0.0] * field_count,
            "entry_label_sensitivity": _uniform(
                rng, search["label_sensitivity_range"]
            ),
            "entry_label_specificity": _uniform(
                rng, search["label_specificity_range"]
            ),
            "persistence_label_sensitivity": _uniform(
                rng, search["label_sensitivity_range"]
            ),
            "persistence_label_specificity": _uniform(
                rng, search["label_specificity_range"]
            ),
            "persistence_censoring": _uniform(
                rng, search["persistence_censoring_range"]
            ),
        }
        scenarios.append(
            _normalize_scenario(
                raw,
                scenario_index=index,
                scenario_kind="adversarial",
                field_count=field_count,
            )
        )
    return scenarios


def _curve_shape(
    grid: np.ndarray,
    *,
    peak: float,
    width: float,
    shape: str,
    secondary_offset: float,
    secondary_weight: float,
) -> np.ndarray:
    if shape == "flat":
        return np.ones_like(grid, dtype=np.float64)
    primary = np.exp(-0.5 * np.square((grid - peak) / width))
    if shape == "bimodal":
        secondary = np.exp(
            -0.5 * np.square((grid - (peak + secondary_offset)) / width)
        )
        primary = primary + secondary_weight * secondary
    maximum = float(np.max(primary))
    require(maximum > 0.0, "curve shape maximum must be positive")
    return primary / maximum


def _observed_probability(probability: np.ndarray, sensitivity: float, specificity: float) -> np.ndarray:
    observed = sensitivity * probability + (1.0 - specificity) * (1.0 - probability)
    return np.clip(observed, 1e-8, 1.0 - 1e-8)


def _earliest_peak(grid: np.ndarray, values: np.ndarray) -> float:
    require(values.size == grid.size, "curve/grid length drift")
    require(np.all(np.isfinite(values)), "truth curve contains nonfinite values")
    maximum = float(np.max(values))
    indices = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-12))
    require(indices.size > 0, "curve has no peak")
    return float(grid[int(indices[0])])


def _component_rng(
    base_seed: int,
    scenario_seed_code: int,
    replicate: int,
    component: int,
) -> np.random.Generator:
    seed = np.random.SeedSequence(
        [base_seed, scenario_seed_code, replicate, component]
    )
    return np.random.default_rng(seed)


def _generate_panel(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    replicate: int,
) -> dict[str, Any]:
    design = config["design"]
    grid = np.asarray(EXPECTED_EVENT_GRID, dtype=np.float64)
    field_count = int(design["field_count"])
    episodes_per_field = int(design["episodes_per_field"])
    rng = _component_rng(
        int(design["base_seed"]),
        int(scenario["scenario_seed_code"]),
        replicate,
        1,
    )
    episodes: list[dict[str, Any]] = []
    for field_id in range(field_count):
        for episode_id in range(episodes_per_field):
            entry_jitter = float(rng.normal(0.0, scenario["episode_peak_sd"]))
            persistence_jitter = float(
                rng.normal(0.0, scenario["episode_peak_sd"])
            )
            latent_entry_peak = (
                scenario["entry_peak"]
                + scenario["field_entry_peak_offsets"][field_id]
                + entry_jitter
            )
            latent_persistence_peak = (
                scenario["persistence_peak"]
                + scenario["field_persistence_peak_offsets"][field_id]
                + persistence_jitter
            )
            observed_entry_peak = latent_entry_peak + scenario["observed_entry_shift"]
            observed_persistence_peak = (
                latent_persistence_peak + scenario["observed_persistence_shift"]
            )
            entry_shape = _curve_shape(
                grid,
                peak=observed_entry_peak,
                width=scenario["entry_width"],
                shape=scenario["entry_shape"],
                secondary_offset=scenario["secondary_peak_offset"],
                secondary_weight=scenario["secondary_peak_weight"],
            )
            persistence_shape = _curve_shape(
                grid,
                peak=observed_persistence_peak,
                width=scenario["persistence_width"],
                shape=scenario["persistence_shape"],
                secondary_offset=scenario["secondary_peak_offset"],
                secondary_weight=scenario["secondary_peak_weight"],
            )
            expected_entry = np.clip(
                scenario["base_entry_probability"]
                + scenario["entry_amplitude"] * entry_shape,
                1e-8,
                0.95,
            )
            expected_persistence = np.clip(
                scenario["base_persistence_probability"]
                + scenario["persistence_amplitude"] * persistence_shape,
                1e-8,
                0.99,
            )
            if scenario["mechanism_family"] == "capacity_depletion":
                expected_persistence = np.clip(
                    expected_persistence
                    - 0.50 * scenario["persistence_amplitude"] * entry_shape,
                    1e-8,
                    0.99,
                )
            censor_scale = np.linspace(0.0, 1.0, grid.size)
            expected_persistence = np.clip(
                expected_persistence
                * (1.0 - scenario["persistence_censoring"] * censor_scale),
                1e-8,
                0.99,
            )
            observed_entry = _observed_probability(
                expected_entry,
                scenario["entry_label_sensitivity"],
                scenario["entry_label_specificity"],
            )
            observed_persistence = _observed_probability(
                expected_persistence,
                scenario["persistence_label_sensitivity"],
                scenario["persistence_label_specificity"],
            )
            opportunities = np.maximum(
                1,
                rng.poisson(float(scenario["opportunity_level"]), size=grid.size),
            ).astype(np.int64)
            entries = rng.binomial(opportunities, observed_entry).astype(np.int64)
            persistent = rng.binomial(entries, observed_persistence).astype(np.int64)
            latent_entry_curve = _curve_shape(
                grid,
                peak=latent_entry_peak,
                width=scenario["entry_width"],
                shape=scenario["entry_shape"],
                secondary_offset=scenario["secondary_peak_offset"],
                secondary_weight=scenario["secondary_peak_weight"],
            )
            latent_persistence_curve = _curve_shape(
                grid,
                peak=latent_persistence_peak,
                width=scenario["persistence_width"],
                shape=scenario["persistence_shape"],
                secondary_offset=scenario["secondary_peak_offset"],
                secondary_weight=scenario["secondary_peak_weight"],
            )
            episodes.append(
                {
                    "field_id": field_id,
                    "episode_id": episode_id,
                    "opportunities": opportunities,
                    "entries": entries,
                    "persistent": persistent,
                    "observable_entry_probability": observed_entry,
                    "observable_persistence_probability": observed_persistence,
                    "latent_delta": _earliest_peak(grid, latent_entry_curve)
                    - _earliest_peak(grid, latent_persistence_curve),
                    "observable_delta": _earliest_peak(grid, observed_entry)
                    - _earliest_peak(grid, observed_persistence),
                }
            )
    return {"grid": grid, "episodes": episodes}


def _kernel_smooth(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    smoothed = np.full_like(values, np.nan, dtype=np.float64)
    for index in range(values.size):
        positions = np.asarray((index - 1, index, index + 1))
        valid_position = (positions >= 0) & (positions < values.size)
        positions = positions[valid_position]
        local_weights = weights[valid_position]
        finite = np.isfinite(values[positions])
        if not np.any(finite):
            continue
        local_weights = local_weights[finite]
        local_weights = local_weights / local_weights.sum()
        smoothed[index] = float(np.dot(values[positions][finite], local_weights))
    return smoothed


def _estimated_episode_delta(
    grid: np.ndarray,
    episode: Mapping[str, Any],
    *,
    estimator: str,
    kernel_weights: np.ndarray,
    minimum_finite: int,
) -> float | None:
    opportunities = np.asarray(episode["opportunities"], dtype=np.float64)
    entries = np.asarray(episode["entries"], dtype=np.float64)
    persistent = np.asarray(episode["persistent"], dtype=np.float64)
    entry_rate = entries / opportunities
    persistence_rate = np.divide(
        persistent,
        entries,
        out=np.full_like(persistent, np.nan),
        where=entries > 0,
    )
    if estimator == "kernel_earliest_argmax":
        entry_rate = _kernel_smooth(entry_rate, kernel_weights)
        persistence_rate = _kernel_smooth(persistence_rate, kernel_weights)
    require(estimator in EXPECTED_ESTIMATORS, f"unsupported estimator: {estimator}")
    if (
        np.isfinite(entry_rate).sum() < minimum_finite
        or np.isfinite(persistence_rate).sum() < minimum_finite
    ):
        return None
    entry_indices = np.flatnonzero(np.isfinite(entry_rate))
    persistence_indices = np.flatnonzero(np.isfinite(persistence_rate))
    entry_values = entry_rate[entry_indices]
    persistence_values = persistence_rate[persistence_indices]
    entry_max = float(np.max(entry_values))
    persistence_max = float(np.max(persistence_values))
    entry_peak_index = int(
        entry_indices[
            np.flatnonzero(
                np.isclose(entry_values, entry_max, rtol=0.0, atol=1e-12)
            )[0]
        ]
    )
    persistence_peak_index = int(
        persistence_indices[
            np.flatnonzero(
                np.isclose(
                    persistence_values,
                    persistence_max,
                    rtol=0.0,
                    atol=1e-12,
                )
            )[0]
        ]
    )
    return float(grid[entry_peak_index] - grid[persistence_peak_index])


def _field_balanced(values: Mapping[int, Sequence[float]]) -> float:
    field_means = [float(np.mean(field_values)) for field_values in values.values()]
    return float(np.mean(field_means))


def _estimate_panel(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    replicate: int,
    panel: Mapping[str, Any],
    *,
    estimator: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    design = config["design"]
    grid = np.asarray(panel["grid"], dtype=np.float64)
    weights = np.asarray(design["kernel_weights"], dtype=np.float64)
    minimum_finite = int(design["minimum_finite_event_times_per_curve"])
    minimum_episodes = int(design["minimum_supported_episodes_per_field"])
    estimated_by_field: dict[int, list[float]] = defaultdict(list)
    truth_by_field: dict[int, list[float]] = defaultdict(list)
    latent_by_field: dict[int, list[float]] = defaultdict(list)
    for episode in panel["episodes"]:
        estimate = _estimated_episode_delta(
            grid,
            episode,
            estimator=estimator,
            kernel_weights=weights,
            minimum_finite=minimum_finite,
        )
        if estimate is not None:
            field_id = int(episode["field_id"])
            estimated_by_field[field_id].append(estimate)
            truth_by_field[field_id].append(float(episode["observable_delta"]))
            latent_by_field[field_id].append(float(episode["latent_delta"]))
    field_count = int(design["field_count"])
    valid = len(estimated_by_field) == field_count and all(
        len(estimated_by_field[field_id]) >= minimum_episodes
        for field_id in range(field_count)
    )
    truth_delta = _field_balanced(truth_by_field)
    latent_delta = _field_balanced(latent_by_field)
    if not valid:
        return {
            "valid": False,
            "point_estimate": None,
            "one_sided_lower": None,
            "two_sided_lower": None,
            "two_sided_upper": None,
            "passed": False,
            "truth_delta": truth_delta,
            "latent_delta": latent_delta,
            "supported_episodes": sum(map(len, estimated_by_field.values())),
            "episode_nonpositive_share": None,
        }
    point = _field_balanced(estimated_by_field)
    bootstrap_rng = _component_rng(
        int(design["base_seed"]),
        int(scenario["scenario_seed_code"]),
        replicate,
        10 + EXPECTED_ESTIMATORS.index(estimator),
    )
    bootstrap_values = np.empty(bootstrap_replicates, dtype=np.float64)
    for bootstrap_index in range(bootstrap_replicates):
        sampled: dict[int, list[float]] = {}
        for field_id in range(field_count):
            field_values = np.asarray(estimated_by_field[field_id], dtype=np.float64)
            indices = bootstrap_rng.integers(
                0,
                field_values.size,
                size=field_values.size,
            )
            sampled[field_id] = field_values[indices].tolist()
        bootstrap_values[bootstrap_index] = _field_balanced(sampled)
    one_sided_lower = float(np.quantile(bootstrap_values, 0.05))
    two_sided_lower = float(np.quantile(bootstrap_values, 0.025))
    two_sided_upper = float(np.quantile(bootstrap_values, 0.975))
    decision = design["decision_rule"]
    passed = bool(
        point >= float(decision["minimum_point_gap_cohort_years"])
        and one_sided_lower
        > float(decision["one_sided_lower_bound_strictly_greater_than"])
    )
    episode_values = np.asarray(
        [value for values in estimated_by_field.values() for value in values],
        dtype=np.float64,
    )
    return {
        "valid": True,
        "point_estimate": point,
        "one_sided_lower": one_sided_lower,
        "two_sided_lower": two_sided_lower,
        "two_sided_upper": two_sided_upper,
        "passed": passed,
        "truth_delta": truth_delta,
        "latent_delta": latent_delta,
        "supported_episodes": int(episode_values.size),
        "episode_nonpositive_share": float(np.mean(episode_values <= 0.0)),
    }


def _flag_estimate(
    result: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    minimum_point_gap: float,
) -> dict[str, bool]:
    valid = bool(result["valid"])
    passed = bool(result["passed"])
    point = result["point_estimate"]
    truth = float(result["truth_delta"])
    identifiable = bool(scenario["truth_identifiable"])
    sign_reversal = bool(
        valid
        and identifiable
        and point is not None
        and ((truth > 0.0 and point < 0.0) or (truth < 0.0 and point > 0.0))
    )
    confident_wrong = bool(
        valid
        and identifiable
        and not (
            float(result["two_sided_lower"])
            <= truth
            <= float(result["two_sided_upper"])
        )
    )
    aggregation_paradox = bool(
        valid
        and point is not None
        and point > 0.0
        and result["episode_nonpositive_share"] is not None
        and float(result["episode_nonpositive_share"]) > 0.5
    )
    return {
        "directional_false_positive": bool(identifiable and truth <= 0.0 and passed),
        "delayed_feedback_false_negative": bool(
            scenario["feedback_present"]
            and identifiable
            and truth >= minimum_point_gap
            and not passed
        ),
        "sign_reversal": sign_reversal,
        "confident_wrong_interval": confident_wrong,
        "nonidentifiable_truth_pass": bool(not identifiable and passed),
        "mechanism_overreach_risk": bool(
            not scenario["feedback_present"] and passed
        ),
        "aggregation_paradox": aggregation_paradox,
        "estimator_disagreement": False,
    }


def _replicate_rows(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    replicate: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    panel = _generate_panel(config, scenario, replicate)
    rows: list[dict[str, Any]] = []
    for estimator in EXPECTED_ESTIMATORS:
        result = _estimate_panel(
            config,
            scenario,
            replicate,
            panel,
            estimator=estimator,
            bootstrap_replicates=bootstrap_replicates,
        )
        flags = _flag_estimate(
            result,
            scenario,
            minimum_point_gap=float(
                config["design"]["decision_rule"][
                    "minimum_point_gap_cohort_years"
                ]
            ),
        )
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_kind": scenario["scenario_kind"],
                "mechanism_family": scenario["mechanism_family"],
                "feedback_present": scenario["feedback_present"],
                "truth_identifiable": scenario["truth_identifiable"],
                "replicate": replicate,
                "estimator": estimator,
                **result,
                "flags": flags,
            }
        )
    raw, kernel = rows
    raw_sign = 0 if raw["point_estimate"] is None else int(np.sign(raw["point_estimate"]))
    kernel_sign = (
        0 if kernel["point_estimate"] is None else int(np.sign(kernel["point_estimate"]))
    )
    disagreement = bool(
        raw["passed"] != kernel["passed"] or raw_sign != kernel_sign
    )
    for row in rows:
        row["flags"]["estimator_disagreement"] = disagreement
        row["surprise_score"] = sum(
            FLAG_WEIGHTS[name] for name, active in row["flags"].items() if active
        )
    return rows


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group in sorted(grouped.items(), key=lambda item: item[0]):
        valid = [row for row in group if row["valid"]]
        identifiable = [row for row in valid if row["truth_identifiable"]]
        biases = [
            float(row["point_estimate"]) - float(row["truth_delta"])
            for row in identifiable
        ]
        coverage = [
            float(row["two_sided_lower"])
            <= float(row["truth_delta"])
            <= float(row["two_sided_upper"])
            for row in identifiable
        ]
        summary = {key: value for key, value in zip(keys, group_key, strict=True)}
        summary.update(
            {
                "replicate_rows": len(group),
                "valid_rate": len(valid) / len(group),
                "pass_rate": sum(bool(row["passed"]) for row in group) / len(group),
                "mean_point_estimate": _safe_mean(
                    [float(row["point_estimate"]) for row in valid]
                ),
                "mean_truth_delta": _safe_mean(
                    [float(row["truth_delta"]) for row in identifiable]
                ),
                "mean_bias": _safe_mean(biases),
                "two_sided_coverage": _safe_mean([float(value) for value in coverage]),
                "flag_rates": {
                    flag: sum(bool(row["flags"][flag]) for row in group) / len(group)
                    for flag in FLAG_WEIGHTS
                },
            }
        )
        summaries.append(summary)
    return summaries


def _counterexamples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["surprise_score"]) > 0]
    selected.sort(
        key=lambda row: (
            -int(row["surprise_score"]),
            str(row["scenario_id"]),
            int(row["replicate"]),
            str(row["estimator"]),
        )
    )
    return [dict(row) for row in selected[:200]]


def plan(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "status": "PASS_SYNTHETIC_RESULT_SPACE_PLAN_NO_EXECUTION",
        "config_sha256": _sha256(config_path),
        "event_time_grid": list(EXPECTED_EVENT_GRID),
        "mechanism_families": list(MECHANISM_FAMILIES),
        "canonical_scenario_count": len(config["canonical_scenarios"]),
        "estimators": list(EXPECTED_ESTIMATORS),
        "profiles": config["design"]["profiles"],
        "source_or_empirical_row_read": "FORBIDDEN",
        "scientific_effect": "NOT_ESTABLISHED",
    }


def run(config_path: Path, output_root: Path, profile_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    profiles = config["design"]["profiles"]
    require(profile_name in profiles, f"unknown profile: {profile_name}")
    profile = profiles[profile_name]
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=False)
    require(stat.S_IMODE(output_root.stat().st_mode) == 0o700, "output root mode drift")

    canonical = list(config["canonical_scenarios"])
    adversarial = _random_scenarios(
        config, int(profile["adversarial_scenarios"])
    )
    scenarios = canonical + adversarial
    manifest_rows = [
        {
            key: value
            for key, value in scenario.items()
            if key not in {"scenario_index"}
        }
        for scenario in scenarios
    ]
    estimate_rows: list[dict[str, Any]] = []
    for scenario_number, scenario in enumerate(scenarios, start=1):
        replicates = (
            int(profile["canonical_replicates"])
            if scenario["scenario_kind"] == "canonical"
            else int(profile["adversarial_replicates"])
        )
        for replicate in range(replicates):
            estimate_rows.extend(
                _replicate_rows(
                    config,
                    scenario,
                    replicate,
                    int(profile["bootstrap_replicates"]),
                )
            )
        if scenario_number % 25 == 0 or scenario_number == len(scenarios):
            print(
                json.dumps(
                    {
                        "status": "RUNNING_SYNTHETIC_ONLY",
                        "completed_scenarios": scenario_number,
                        "total_scenarios": len(scenarios),
                        "estimate_rows": len(estimate_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    counterexamples = _counterexamples(estimate_rows)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_SYNTHETIC_RESULT_SPACE_EXECUTION_COUNTEREXAMPLES_EXPECTED_NO_EMPIRICAL_ROWS",
        "completed_at_utc": _utc_now(),
        "profile": profile_name,
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "design": {
            "event_time_grid": list(EXPECTED_EVENT_GRID),
            "field_count": config["design"]["field_count"],
            "episodes_per_field": config["design"]["episodes_per_field"],
            "estimators": list(EXPECTED_ESTIMATORS),
            "mechanism_families": list(MECHANISM_FAMILIES),
            "profile_counts": profile,
        },
        "execution_counts": {
            "canonical_scenarios": len(canonical),
            "adversarial_scenarios": len(adversarial),
            "total_scenarios": len(scenarios),
            "replicate_estimate_rows": len(estimate_rows),
            "counterexample_rows_total": sum(
                int(row["surprise_score"]) > 0 for row in estimate_rows
            ),
            "counterexample_rows_retained": len(counterexamples),
        },
        "canonical_operating_characteristics": _summaries(
            [row for row in estimate_rows if row["scenario_kind"] == "canonical"],
            keys=("scenario_id", "mechanism_family", "estimator"),
        ),
        "mechanism_operating_characteristics": _summaries(
            estimate_rows,
            keys=("scenario_kind", "mechanism_family", "estimator"),
        ),
        "flag_counts": {
            flag: sum(bool(row["flags"][flag]) for row in estimate_rows)
            for flag in FLAG_WEIGHTS
        },
        "access_accounting": {
            "source_or_empirical_rows_read": 0,
            "current_g1_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "models_fitted_to_empirical_data": 0,
        },
        "authorization": {
            "synthetic_design_validation": "COMPLETE_FOR_PROFILE",
            "empirical_parameter_calibration": "FORBIDDEN_NOT_RUN",
            "confirmatory_execution": "FORBIDDEN_NOT_RUN",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": config["claim_boundary"],
    }

    manifest_path = output_root / "scenario_manifest.jsonl"
    estimates_path = output_root / "replicate_estimates.jsonl"
    counterexamples_path = output_root / "counterexamples.json"
    report_path = output_root / "result_space_report.json"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(estimates_path, estimate_rows)
    _write_json(counterexamples_path, counterexamples)
    _write_json(report_path, report)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS_RECEIPT_WRITTEN_LAST_SYNTHETIC_ONLY",
        "completed_at_utc": _utc_now(),
        "profile": profile_name,
        "config": {
            "relative_path": config_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "producer": {
            "relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "artifacts": [
            {
                "basename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (
                manifest_path,
                estimates_path,
                counterexamples_path,
                report_path,
            )
        ],
        "access_accounting": report["access_accounting"],
        "scientific_effect": "NOT_ESTABLISHED",
    }
    receipt_path = output_root / "receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": report["status"],
        "output_root": str(output_root),
        "profile": profile_name,
        "scenario_count": len(scenarios),
        "replicate_estimate_rows": len(estimate_rows),
        "counterexample_rows_total": report["execution_counts"][
            "counterexample_rows_total"
        ],
        "receipt_sha256": _sha256(receipt_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        require(args.output_root is None, "plan command forbids --output-root")
        result = plan(args.config)
    else:
        require(args.output_root is not None, "run command requires --output-root")
        result = run(args.config, args.output_root, args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
