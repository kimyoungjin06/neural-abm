#!/usr/bin/env python3
"""Run public nonblind stochastic set development without opened challenge data."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_lean_h3_stochastic_set_development.v3"
EXPECTED_MODE = "PUBLIC_NONBLIND_DISJOINT_SYNTHETIC_DEVELOPMENT_NO_OPENED_CHALLENGE"
EXPECTED_STATUS = "PRECOMMIT_REQUIRED_NO_GENERATION_NO_RESULT_NO_SELECTION"
EXPECTED_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0)
CANDIDATE = "jeffreys_normalized_regret_field_vector_v3"
TRACKED_PATHS = (
    "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml",
    "scripts/develop_pivot_feedback_lean_h3_stochastic_set_v3.py",
    "scripts/verify_pivot_feedback_lean_h3_stochastic_set_v3.py",
    "tests/test_pivot_feedback_lean_h3_stochastic_set_v3.py",
)


class StochasticSetDevelopmentError(RuntimeError):
    """Raised when the public development contract or artifact chain drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StochasticSetDevelopmentError(message)


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
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
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


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root must be mapping: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            require(isinstance(payload, dict), f"invalid row {line_number}: {path}")
            rows.append(payload)
    require(rows, f"empty JSONL: {path}")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be positive integer",
    )
    return int(value)


def load_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("mode") == EXPECTED_MODE, "config mode drift")
    require(config.get("status") == EXPECTED_STATUS, "config status drift")
    predecessor = _mapping(config.get("predecessor"), "predecessor")
    predecessor_path = ROOT / str(predecessor.get("summary_relative_path"))
    require(_sha256(predecessor_path) == predecessor.get("summary_sha256"), "predecessor drift")
    scientific = _mapping(config.get("scientific_object"), "scientific object")
    require(
        tuple(float(value) for value in scientific.get("event_time_grid", ()))
        == EXPECTED_GRID,
        "event grid drift",
    )
    audit = _mapping(config.get("scale_audit"), "scale audit")
    require(
        audit.get("selection_criterion")
        == "POSITIVE_AFFINE_RATE_TRANSFORMATION_INVARIANCE",
        "scale criterion drift",
    )
    retained = _mapping(audit.get("retained_rule"), "retained scale rule")
    require(math.isclose(float(retained.get("inclusive_regret_margin")), 0.10), "regret margin drift")
    absolute = _mapping(audit.get("retained_absolute_rule"), "absolute floor")
    require(
        math.isclose(
            float(absolute.get("inclusive_probability_prominence_floor")),
            0.01,
        ),
        "flat floor drift",
    )
    candidate = _mapping(config.get("stochastic_candidate"), "candidate")
    require(candidate.get("name") == CANDIDATE, "candidate drift")
    posterior = _mapping(candidate.get("posterior_peak_set"), "posterior peak set")
    require(posterior.get("earliest_argmax_tie_break") == "FORBIDDEN", "tie drift")
    aggregation = _mapping(candidate.get("field_aggregation"), "aggregation")
    require(aggregation.get("preserve_field_vector") is True, "vector drift")
    require(aggregation.get("binary_pass_fail_decision") == "FORBIDDEN", "decision drift")
    generator = _mapping(config.get("public_development_generator"), "generator")
    require(generator.get("field_count") == 4, "field count drift")
    require(generator.get("episodes_per_field") == 8, "episode count drift")
    templates = _mapping(generator.get("curve_templates"), "curve templates")
    for name, curve in templates.items():
        require(isinstance(name, str), "curve name drift")
        require(isinstance(curve, list) and len(curve) == 5, f"curve length: {name}")
        require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0.0 <= float(value) <= 1.0
                for value in curve
            ),
            f"curve value drift: {name}",
        )
    scenarios = generator.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 16, "scenario count drift")
    seen: set[str] = set()
    for scenario in scenarios:
        require(isinstance(scenario, dict), "scenario must be mapping")
        scenario_id = str(scenario.get("scenario_id"))
        require(scenario_id not in seen, f"duplicate scenario: {scenario_id}")
        seen.add(scenario_id)
        pairs = scenario.get("field_pairs")
        require(isinstance(pairs, list) and len(pairs) == 4, "field pair drift")
        for pair in pairs:
            require(
                isinstance(pair, list)
                and len(pair) == 2
                and pair[0] in templates
                and pair[1] in templates,
                f"curve reference drift: {scenario_id}",
            )
        _positive_int(scenario.get("opportunities"), f"opportunities {scenario_id}")
    profiles = _mapping(generator.get("profiles"), "profiles")
    for name, profile in profiles.items():
        profile_map = _mapping(profile, f"profile {name}")
        _positive_int(profile_map.get("replicates_per_scenario"), f"{name}.replicates")
        _positive_int(profile_map.get("posterior_draws"), f"{name}.draws")
    reporting = _mapping(config.get("development_reporting"), "reporting")
    require(reporting.get("thresholds") == "NONE", "threshold drift")
    authorization = _mapping(config.get("authorization"), "authorization")
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
        require(authorization.get(key) == "FORBIDDEN", f"authorization drift: {key}")
    require(authorization.get("candidate_admissibility") == "NOT_EVALUATED", "admissibility drift")
    require(authorization.get("candidate_selected") == "NO", "candidate selection drift")
    require(authorization.get("estimator_selected") == "NO", "estimator selection drift")
    require(authorization.get("support_rule_selected") == "NO", "support selection drift")
    require(authorization.get("fresh_blind_authorized") is False, "blind authorization drift")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")
    return config


def _verify_precommit(precommit: str) -> None:
    require(len(precommit) == 40, "precommit must be full SHA-1")
    subprocess.run(
        ["git", "cat-file", "-e", f"{precommit}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for relative in TRACKED_PATHS:
        committed = subprocess.run(
            ["git", "show", f"{precommit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        require(committed == (ROOT / relative).read_bytes(), f"post-precommit drift: {relative}")


def _rng(config: Mapping[str, Any], *parts: Any) -> np.random.Generator:
    seed = config["public_development_generator"]["deterministic_seed"]
    material = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(np.random.SeedSequence(list(digest)))


def _normalized_peak_set(
    curve: Sequence[float | None],
    *,
    regret_margin: float,
    flat_floor: float,
) -> dict[str, Any]:
    if any(value is None for value in curve):
        return {"valid": False, "reason": "MISSING_REGISTERED_TIME"}
    values = np.asarray(curve, dtype=np.float64)
    prominence = float(np.max(values) - np.min(values))
    if prominence <= flat_floor + 1e-12:
        return {"valid": False, "reason": "PRACTICALLY_FLAT_CURVE"}
    regret = (float(np.max(values)) - values) / prominence
    indices = np.flatnonzero(regret <= regret_margin + 1e-12).tolist()
    if 0 in indices or len(EXPECTED_GRID) - 1 in indices:
        return {"valid": False, "reason": "BOUNDARY_PRACTICAL_PEAK_SET"}
    peak_set = [EXPECTED_GRID[index] for index in indices]
    contiguous = indices == list(range(indices[0], indices[-1] + 1))
    return {
        "valid": True,
        "reason": None,
        "peak_set": peak_set,
        "multimodal": len(indices) > 1 and not contiguous,
    }


def _episode_gap_truth(
    entry: Sequence[float | None],
    persistence: Sequence[float | None],
    *,
    regret_margin: float,
    flat_floor: float,
) -> dict[str, Any]:
    entry_set = _normalized_peak_set(entry, regret_margin=regret_margin, flat_floor=flat_floor)
    persistence_set = _normalized_peak_set(
        persistence,
        regret_margin=regret_margin,
        flat_floor=flat_floor,
    )
    reasons = sorted(
        {
            str(result["reason"])
            for result in (entry_set, persistence_set)
            if not result["valid"]
        }
    )
    if reasons:
        return {"valid": False, "reasons": reasons}
    gaps = sorted(
        {
            float(entry_value) - float(persistence_value)
            for entry_value in entry_set["peak_set"]
            for persistence_value in persistence_set["peak_set"]
        }
    )
    return {
        "valid": True,
        "reasons": [],
        "interval": [min(gaps), max(gaps)],
        "multimodal": bool(entry_set["multimodal"] or persistence_set["multimodal"]),
    }


def _aggregate_episode_results(
    config: Mapping[str, Any],
    scenario_id: str,
    group: str,
    replicate: int,
    episode_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregation = config["stochastic_candidate"]["field_aggregation"]
    field_count = int(aggregation["field_count"])
    minimum = int(aggregation["minimum_valid_episodes_per_field"])
    field_intervals: list[list[float]] = []
    invalid_reasons: Counter[str] = Counter()
    any_set = False
    any_multimodal = False
    for field_id in range(field_count):
        rows = [row for row in episode_results if int(row["field_id"]) == field_id]
        valid = [row for row in rows if bool(row["valid"])]
        for row in rows:
            invalid_reasons.update(row.get("reasons", []))
        if len(valid) < minimum:
            invalid_reasons["INSUFFICIENT_VALID_EPISODES_PER_FIELD"] += 1
            continue
        lower = float(np.mean([row["interval"][0] for row in valid]))
        upper = float(np.mean([row["interval"][1] for row in valid]))
        field_intervals.append([lower, upper])
        any_set = any_set or not math.isclose(lower, upper)
        any_multimodal = any_multimodal or any(bool(row["multimodal"]) for row in valid)
    if len(field_intervals) != field_count:
        return {
            "scenario_id": scenario_id,
            "group": group,
            "replicate": replicate,
            "valid": False,
            "field_intervals": None,
            "field_balanced_interval_hull": None,
            "identification_state": "INVALID",
            "direction_state": "INVALID_NO_DECISION",
            "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        }
    lower = float(np.mean([interval[0] for interval in field_intervals]))
    upper = float(np.mean([interval[1] for interval in field_intervals]))
    floor = float(
        config["stochastic_candidate"]["descriptive_direction_state"][
            "meaningful_positive_floor_cohort_years"
        ]
    )
    if all(interval[0] >= floor for interval in field_intervals):
        direction = "UNANIMOUS_MEANINGFUL_POSITIVE"
    elif all(interval[1] <= 0.0 for interval in field_intervals):
        direction = "UNANIMOUS_NONPOSITIVE"
    else:
        direction = "MIXED_OR_PARTIALLY_IDENTIFIED_NO_BINARY_DECISION"
    if any_multimodal:
        state = "SET_IDENTIFIED_MULTIMODAL_FIELD_VECTOR"
    elif any_set:
        state = "SET_IDENTIFIED_FIELD_VECTOR"
    else:
        state = "POINT_IDENTIFIED_FIELD_VECTOR"
    return {
        "scenario_id": scenario_id,
        "group": group,
        "replicate": replicate,
        "valid": True,
        "field_intervals": field_intervals,
        "field_balanced_interval_hull": [lower, upper],
        "identification_state": state,
        "direction_state": direction,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
    }


def _scale_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    audit = config["scale_audit"]
    fixture = audit["analytic_fixture"]
    base = np.asarray(fixture["base_curve"], dtype=np.float64)
    transformed = float(fixture["affine_scale"]) * base + float(fixture["affine_shift"])
    absolute_margin = float(audit["rejected_rule"]["margin"])
    regret_margin = float(audit["retained_rule"]["inclusive_regret_margin"])

    def absolute_indices(values: np.ndarray) -> list[int]:
        return np.flatnonzero(np.max(values) - values <= absolute_margin + 1e-12).tolist()

    def normalized_indices(values: np.ndarray) -> list[int]:
        prominence = np.max(values) - np.min(values)
        regret = (np.max(values) - values) / prominence
        return np.flatnonzero(regret <= regret_margin + 1e-12).tolist()

    absolute_invariant = absolute_indices(base) == absolute_indices(transformed)
    normalized_invariant = normalized_indices(base) == normalized_indices(transformed)
    require(
        absolute_invariant is bool(fixture["expected_absolute_rule_invariant"]),
        "absolute scale audit drift",
    )
    require(
        normalized_invariant is bool(fixture["expected_normalized_rule_invariant"]),
        "normalized scale audit drift",
    )
    return {
        "selection_criterion": audit["selection_criterion"],
        "absolute_rule_invariant": absolute_invariant,
        "normalized_rule_invariant": normalized_invariant,
        "retained_rule": audit["retained_rule"]["name"],
        "status": "PASS_ANALYTIC_SCALE_INVARIANCE_AUDIT",
    }


def plan(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "status": "PASS_STOCHASTIC_SET_DEVELOPMENT_PLAN_NO_GENERATION",
        "config_sha256": _sha256(config_path),
        "candidate": CANDIDATE,
        "scenario_count": len(config["public_development_generator"]["scenarios"]),
        "profiles": config["public_development_generator"]["profiles"],
        "scale_audit": _scale_audit(config),
        "opened_challenge_rows_read": 0,
        "candidate_selected": "NO",
        "fresh_blind_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def generate(
    config_path: Path,
    output_root: Path,
    profile_name: str,
    precommit: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    _verify_precommit(precommit)
    generator = config["public_development_generator"]
    profiles = generator["profiles"]
    require(profile_name in profiles, f"unknown profile: {profile_name}")
    require(not output_root.exists(), f"output root exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=False)
    templates = generator["curve_templates"]
    profile = profiles[profile_name]
    replicates = int(profile["replicates_per_scenario"])
    regret_margin = float(config["scale_audit"]["retained_rule"]["inclusive_regret_margin"])
    flat_floor = float(
        config["scale_audit"]["retained_absolute_rule"][
            "inclusive_probability_prominence_floor"
        ]
    )
    panel_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for scenario in generator["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        group = str(scenario["group"])
        for replicate in range(replicates):
            rng = _rng(config, "generate", scenario_id, replicate)
            episodes: list[dict[str, Any]] = []
            truths: list[dict[str, Any]] = []
            for field_id, pair in enumerate(scenario["field_pairs"]):
                for episode_id in range(int(generator["episodes_per_field"])):
                    entry_base = np.asarray(templates[pair[0]], dtype=np.float64)
                    persistence_base = np.asarray(templates[pair[1]], dtype=np.float64)
                    concentration = scenario.get("beta_concentration")
                    if concentration is None:
                        entry_probability = entry_base.copy()
                        persistence_probability = persistence_base.copy()
                    else:
                        kappa = float(concentration)
                        entry_probability = rng.beta(
                            entry_base * kappa,
                            (1.0 - entry_base) * kappa,
                        )
                        persistence_probability = rng.beta(
                            persistence_base * kappa,
                            (1.0 - persistence_base) * kappa,
                        )
                    opportunities = np.full(5, int(scenario["opportunities"]), dtype=np.int64)
                    missing_index = scenario.get("missing_time_index")
                    entry_truth: list[float | None] = entry_probability.tolist()
                    persistence_truth: list[float | None] = persistence_probability.tolist()
                    if missing_index is not None:
                        position = int(missing_index)
                        opportunities[position] = 0
                        entry_truth[position] = None
                        persistence_truth[position] = None
                    entries = rng.binomial(opportunities, entry_probability).astype(np.int64)
                    persistent = rng.binomial(entries, persistence_probability).astype(np.int64)
                    episodes.append(
                        {
                            "field_id": field_id,
                            "episode_id": episode_id,
                            "opportunities": opportunities.tolist(),
                            "entries": entries.tolist(),
                            "persistent": persistent.tolist(),
                        }
                    )
                    truth = _episode_gap_truth(
                        entry_truth,
                        persistence_truth,
                        regret_margin=regret_margin,
                        flat_floor=flat_floor,
                    )
                    truths.append({"field_id": field_id, "episode_id": episode_id, **truth})
            panel_rows.append(
                {
                    "scenario_id": scenario_id,
                    "group": group,
                    "replicate": replicate,
                    "grid": list(EXPECTED_GRID),
                    "episodes": episodes,
                }
            )
            truth_rows.append(
                _aggregate_episode_results(
                    config,
                    scenario_id,
                    group,
                    replicate,
                    truths,
                )
            )
    panels_path = output_root / "public_panels.jsonl"
    truth_path = output_root / "public_truth.jsonl"
    _write_jsonl(panels_path, panel_rows)
    _write_jsonl(truth_path, truth_rows)
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_stochastic_set_generation_receipt.v3",
        "status": "PASS_PUBLIC_NONBLIND_DISJOINT_GENERATION",
        "completed_at_utc": _utc_now(),
        "profile": profile_name,
        "precommit": precommit,
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "scale_audit": _scale_audit(config),
        "panels": {"rows": len(panel_rows), "sha256": _sha256(panels_path)},
        "truth": {"rows": len(truth_rows), "sha256": _sha256(truth_path)},
        "opened_challenge_rows_read": 0,
        "source_or_empirical_rows_read": 0,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    receipt_path = output_root / "generation_receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output_root": str(output_root),
        "panel_rows": len(panel_rows),
        "truth_rows": len(truth_rows),
        "generation_receipt_sha256": _sha256(receipt_path),
    }


def _posterior_peak_set(
    successes: Sequence[int],
    trials: Sequence[int],
    *,
    draws: int,
    rng: np.random.Generator,
    regret_margin: float,
    inclusion_probability: float,
    flat_floor: float,
    flat_probability: float,
) -> dict[str, Any]:
    success = np.asarray(successes, dtype=np.int64)
    trial = np.asarray(trials, dtype=np.int64)
    if np.any(trial <= 0):
        return {"valid": False, "reason": "MISSING_REGISTERED_DENOMINATOR"}
    require(np.all((0 <= success) & (success <= trial)), "count range drift")
    probabilities = rng.beta(
        success[np.newaxis, :] + 0.5,
        trial[np.newaxis, :] - success[np.newaxis, :] + 0.5,
        size=(draws, 5),
    )
    maxima = np.max(probabilities, axis=1)
    minima = np.min(probabilities, axis=1)
    prominence = maxima - minima
    if float(np.mean(prominence <= flat_floor)) >= flat_probability:
        return {"valid": False, "reason": "POSTERIOR_PRACTICALLY_FLAT_CURVE"}
    denominator = np.maximum(prominence, np.finfo(np.float64).eps)
    regret = (maxima[:, np.newaxis] - probabilities) / denominator[:, np.newaxis]
    inclusion = np.mean(regret <= regret_margin, axis=0)
    indices = np.flatnonzero(inclusion >= inclusion_probability).tolist()
    if not indices:
        return {"valid": False, "reason": "UNRESOLVED_POSTERIOR_PEAK_SET"}
    if 0 in indices or 4 in indices:
        return {"valid": False, "reason": "BOUNDARY_POSTERIOR_PEAK_SET"}
    peak_set = [EXPECTED_GRID[index] for index in indices]
    contiguous = indices == list(range(indices[0], indices[-1] + 1))
    return {
        "valid": True,
        "reason": None,
        "peak_set": peak_set,
        "multimodal": len(indices) > 1 and not contiguous,
    }


def _estimate_episode(
    config: Mapping[str, Any],
    scenario_id: str,
    replicate: int,
    episode: Mapping[str, Any],
    draws: int,
) -> dict[str, Any]:
    posterior = config["stochastic_candidate"]["posterior_peak_set"]
    entry = _posterior_peak_set(
        episode["entries"],
        episode["opportunities"],
        draws=draws,
        rng=_rng(config, "estimate", scenario_id, replicate, episode["field_id"], episode["episode_id"], "entry"),
        regret_margin=float(posterior["normalized_regret_margin"]),
        inclusion_probability=float(posterior["minimum_time_inclusion_probability"]),
        flat_floor=float(posterior["practically_flat_prominence_floor"]),
        flat_probability=float(posterior["minimum_flat_posterior_probability_for_invalid"]),
    )
    persistence = _posterior_peak_set(
        episode["persistent"],
        episode["entries"],
        draws=draws,
        rng=_rng(config, "estimate", scenario_id, replicate, episode["field_id"], episode["episode_id"], "persistence"),
        regret_margin=float(posterior["normalized_regret_margin"]),
        inclusion_probability=float(posterior["minimum_time_inclusion_probability"]),
        flat_floor=float(posterior["practically_flat_prominence_floor"]),
        flat_probability=float(posterior["minimum_flat_posterior_probability_for_invalid"]),
    )
    reasons = sorted(
        {
            str(result["reason"])
            for result in (entry, persistence)
            if not bool(result["valid"])
        }
    )
    if reasons:
        return {
            "field_id": int(episode["field_id"]),
            "episode_id": int(episode["episode_id"]),
            "valid": False,
            "reasons": reasons,
        }
    gaps = sorted(
        {
            float(entry_value) - float(persistence_value)
            for entry_value in entry["peak_set"]
            for persistence_value in persistence["peak_set"]
        }
    )
    return {
        "field_id": int(episode["field_id"]),
        "episode_id": int(episode["episode_id"]),
        "valid": True,
        "reasons": [],
        "interval": [min(gaps), max(gaps)],
        "multimodal": bool(entry["multimodal"] or persistence["multimodal"]),
    }


def estimate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    panels_path = output_root / "public_panels.jsonl"
    generation_path = output_root / "generation_receipt.json"
    estimates_path = output_root / "candidate_estimates.jsonl"
    receipt_path = output_root / "estimation_receipt.json"
    require(not estimates_path.exists() and not receipt_path.exists(), "estimation exists")
    generation = _read_json(generation_path)
    require(generation.get("status") == "PASS_PUBLIC_NONBLIND_DISJOINT_GENERATION", "generation incomplete")
    require(generation.get("config_sha256") == _sha256(config_path), "config drift")
    require(
        generation.get("producer_sha256") == _sha256(Path(__file__).resolve()),
        "producer drift",
    )
    require(generation["panels"].get("sha256") == _sha256(panels_path), "panel drift")
    profile = config["public_development_generator"]["profiles"][generation["profile"]]
    draws = int(profile["posterior_draws"])
    rows: list[dict[str, Any]] = []
    for panel in _read_jsonl(panels_path):
        episode_results = [
            _estimate_episode(
                config,
                str(panel["scenario_id"]),
                int(panel["replicate"]),
                episode,
                draws,
            )
            for episode in panel["episodes"]
        ]
        rows.append(
            _aggregate_episode_results(
                config,
                str(panel["scenario_id"]),
                str(panel["group"]),
                int(panel["replicate"]),
                episode_results,
            )
        )
    _write_jsonl(estimates_path, rows)
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_stochastic_set_estimation_receipt.v3",
        "status": "PASS_PUBLIC_PANEL_ESTIMATION_NO_TRUTH_READ",
        "completed_at_utc": _utc_now(),
        "profile": generation["profile"],
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "generation_receipt_sha256": _sha256(generation_path),
        "panels_sha256": _sha256(panels_path),
        "estimates": {"rows": len(rows), "sha256": _sha256(estimates_path)},
        "truth_files_read": 0,
        "opened_challenge_rows_read": 0,
        "source_or_empirical_rows_read": 0,
        "candidate_selected": "NO",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "estimate_rows": len(rows),
        "valid_rows": sum(bool(row["valid"]) for row in rows),
        "estimation_receipt_sha256": _sha256(receipt_path),
        "truth_files_read": 0,
    }


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth_valid = [row for row in rows if bool(row["truth_valid"])]
    truth_invalid = [row for row in rows if not bool(row["truth_valid"])]
    jointly_valid = [row for row in truth_valid if bool(row["valid"])]

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    containment = []
    state_matches = []
    direction_matches = []
    for row in jointly_valid:
        estimated = row["field_intervals"]
        truth = row["truth_field_intervals"]
        containment.append(
            all(
                float(est[0]) <= float(target[0])
                and float(est[1]) >= float(target[1])
                for est, target in zip(estimated, truth)
            )
        )
        state_matches.append(row["identification_state"] == row["truth_identification_state"])
        direction_matches.append(row["direction_state"] == row["truth_direction_state"])
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(row["invalid_reason_counts"])
    return {
        "rows": len(rows),
        "truth_valid_estimator_valid_rate": rate(len(jointly_valid), len(truth_valid)),
        "truth_invalid_estimator_invalid_rate": rate(
            sum(not bool(row["valid"]) for row in truth_invalid),
            len(truth_invalid),
        ),
        "field_vector_interval_containment_among_jointly_valid": rate(
            sum(containment),
            len(containment),
        ),
        "identification_state_match_among_jointly_valid": rate(
            sum(state_matches),
            len(state_matches),
        ),
        "direction_state_match_among_jointly_valid": rate(
            sum(direction_matches),
            len(direction_matches),
        ),
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "denominators": {
            "truth_valid": len(truth_valid),
            "truth_invalid": len(truth_invalid),
            "jointly_valid": len(jointly_valid),
        },
    }


def evaluate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    truth_path = output_root / "public_truth.jsonl"
    estimates_path = output_root / "candidate_estimates.jsonl"
    estimation_path = output_root / "estimation_receipt.json"
    report_path = output_root / "development_report.json"
    receipt_path = output_root / "evaluation_receipt.json"
    require(not report_path.exists() and not receipt_path.exists(), "evaluation exists")
    estimation = _read_json(estimation_path)
    require(estimation.get("status") == "PASS_PUBLIC_PANEL_ESTIMATION_NO_TRUTH_READ", "estimation incomplete")
    require(estimation.get("truth_files_read") == 0, "truth access order drift")
    truth_rows = _read_jsonl(truth_path)
    estimates = _read_jsonl(estimates_path)
    truth_map = {
        (str(row["scenario_id"]), int(row["replicate"])): row for row in truth_rows
    }
    require(len(truth_map) == len(truth_rows), "duplicate truth rows")
    joined: list[dict[str, Any]] = []
    for estimate_row in estimates:
        key = (str(estimate_row["scenario_id"]), int(estimate_row["replicate"]))
        require(key in truth_map, f"missing truth: {key}")
        truth = truth_map[key]
        joined.append(
            {
                **estimate_row,
                "truth_valid": truth["valid"],
                "truth_field_intervals": truth["field_intervals"],
                "truth_identification_state": truth["identification_state"],
                "truth_direction_state": truth["direction_state"],
            }
        )
    require(len(joined) == len(truth_rows), "estimate/truth cardinality drift")
    overall = _metric_block(joined)
    by_group = {
        group: _metric_block([row for row in joined if row["group"] == group])
        for group in sorted({str(row["group"]) for row in joined})
    }
    by_scenario = {
        scenario_id: _metric_block(
            [row for row in joined if row["scenario_id"] == scenario_id]
        )
        for scenario_id in sorted({str(row["scenario_id"]) for row in joined})
    }
    report = {
        "schema_version": "pivot_feedback_lean_h3_stochastic_set_development_report.v3",
        "status": "COMPLETE_PUBLIC_NONBLIND_DEVELOPMENT_DIAGNOSTIC_NO_ADMISSIBILITY",
        "completed_at_utc": _utc_now(),
        "candidate": CANDIDATE,
        "config_sha256": _sha256(config_path),
        "scale_audit": _scale_audit(config),
        "metrics": overall,
        "group_metrics": by_group,
        "scenario_metrics": by_scenario,
        "dependence_boundary": config["dependence_boundary"],
        "disposition": {
            "candidate_admissibility": "NOT_EVALUATED",
            "candidate_selected": "NO",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "fresh_blind_authorized": False,
            "opened_challenge_reuse": "FORBIDDEN",
        },
        "access_accounting": {
            "opened_challenge_panel_rows_read": 0,
            "opened_challenge_private_label_rows_read": 0,
            "source_or_empirical_rows_read": 0,
            "current_g1_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "models_fitted_to_empirical_data": 0,
        },
        "scientific_effect": "NOT_ESTABLISHED",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(report_path, report)
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_stochastic_set_evaluation_receipt.v3",
        "status": "PASS_DESCRIPTIVE_EVALUATION_RECEIPT_NO_SELECTION",
        "completed_at_utc": _utc_now(),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "estimation_receipt_sha256": _sha256(estimation_path),
            "truth_sha256": _sha256(truth_path),
            "estimates_sha256": _sha256(estimates_path),
            "report_sha256": _sha256(report_path),
        },
        "candidate_admissibility": "NOT_EVALUATED",
        "candidate_selected": "NO",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "fresh_blind_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    _write_json(receipt_path, receipt)
    return {
        "status": report["status"],
        "report_sha256": _sha256(report_path),
        "evaluation_receipt_sha256": _sha256(receipt_path),
        "candidate_selected": "NO",
        "fresh_blind_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "generate", "estimate", "evaluate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--precommit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        require(args.output_root is None and args.precommit is None, "plan argument drift")
        result = plan(args.config)
    else:
        require(args.output_root is not None, f"{args.command} requires output root")
        if args.command == "generate":
            require(args.precommit is not None, "generate requires precommit")
            result = generate(args.config, args.output_root, args.profile, args.precommit)
        elif args.command == "estimate":
            require(args.precommit is None, "estimate forbids precommit")
            result = estimate(args.config, args.output_root)
        else:
            require(args.precommit is None, "evaluate forbids precommit")
            result = evaluate(args.config, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
