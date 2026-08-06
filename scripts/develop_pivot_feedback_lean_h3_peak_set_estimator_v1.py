#!/usr/bin/env python3
"""Develop an uncertainty-aware peak-set estimator on an opened synthetic challenge."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    "experiments/configs/pivot_feedback_lean_h3_peak_set_development_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_lean_h3_peak_set_development.v1"
EXPECTED_MODE = "SYNTHETIC_ONLY_OPENED_CHALLENGE_DISCOVERY_DIAGNOSTIC_NOT_ADMISSIBILITY"
EXPECTED_STATUS = "AUTHORIZED_DEVELOPMENT_ONLY_NO_ESTIMATOR_OR_SUPPORT_SELECTION"
EXPECTED_GRID = (-2, -1, 0, 1, 2)
CANDIDATE_NAME = "jeffreys_peak_set_field_balanced_v1"
PANEL_KEYS = {"challenge_id", "replicate", "grid", "episodes"}
EPISODE_KEYS = {
    "field_id",
    "episode_id",
    "opportunities",
    "entries",
    "persistent",
}


class PeakSetDevelopmentError(RuntimeError):
    """Raised when the development-only estimator contract cannot be preserved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PeakSetDevelopmentError(message)


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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
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
    require(path.is_file() and not path.is_symlink(), f"invalid JSON input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root must be mapping: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            require(isinstance(payload, dict), f"invalid row {line_number}: {path}")
            rows.append(payload)
    require(rows, f"empty JSONL input: {path}")
    return rows


def _finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def _positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def load_config(
    path: Path,
    *,
    source_binding_scope: str = "all",
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("mode") == EXPECTED_MODE, "config mode drift")
    require(config.get("status") == EXPECTED_STATUS, "config status drift")
    scientific = config.get("scientific_object")
    require(isinstance(scientific, dict), "scientific object missing")
    require(tuple(scientific.get("event_time_grid", ())) == EXPECTED_GRID, "grid drift")
    candidate = config.get("candidate")
    require(isinstance(candidate, dict), "candidate missing")
    require(candidate.get("name") == CANDIDATE_NAME, "candidate name drift")
    posterior = candidate.get("posterior_peak")
    require(isinstance(posterior, dict), "posterior peak contract missing")
    mass = _finite(posterior.get("peak_set_credible_mass"), "peak set mass")
    modal = _finite(
        posterior.get("minimum_modal_peak_probability"), "modal probability"
    )
    require(0.5 < mass < 1.0, "peak set mass outside (0.5,1)")
    require(0.5 < modal < 1.0, "modal probability outside (0.5,1)")
    _positive_int(
        posterior.get("maximum_peak_set_grid_span"), "maximum peak set span"
    )
    aggregation = candidate.get("aggregation")
    require(isinstance(aggregation, dict), "aggregation missing")
    field_count = _positive_int(aggregation.get("field_count"), "field count")
    episodes = _positive_int(
        aggregation.get("episodes_per_field"), "episodes per field"
    )
    minimum = _positive_int(
        aggregation.get("minimum_valid_episodes_per_field"),
        "minimum valid episodes",
    )
    require(minimum <= episodes, "minimum valid episodes exceed total")
    require(field_count == 4 and episodes == 8, "opened challenge dimension drift")
    profiles = candidate.get("profiles")
    require(isinstance(profiles, dict) and profiles, "profiles missing")
    for name, profile in profiles.items():
        require(isinstance(name, str) and isinstance(profile, dict), "invalid profile")
        _positive_int(profile.get("posterior_draws"), f"{name}.posterior_draws")
    source = config.get("source_diagnostic")
    require(isinstance(source, dict), "source diagnostic missing")
    source_root = Path(str(source.get("root")))
    bindings = {
        "challenge_panels_sha256": source_root / "challenge_panels.jsonl",
        "private_labels_sha256": source_root / "private_labels.json",
        "frozen_blind_estimates_sha256": source_root / "blind_estimates.jsonl",
        "frozen_unblinded_report_sha256": source_root / "unblinded_report.json",
    }
    require(
        source_binding_scope in {"none", "panels", "all"},
        "invalid source binding scope",
    )
    keys_to_verify = {
        "none": (),
        "panels": ("challenge_panels_sha256",),
        "all": tuple(bindings),
    }[source_binding_scope]
    for key in keys_to_verify:
        require(
            _sha256(bindings[key]) == source.get(key),
            f"source binding drift: {key}",
        )
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
    require(authorization.get("estimator_selected") == "NO", "selection drift")
    require(authorization.get("support_rule_selected") == "NO", "support drift")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")
    return config


def _rng(
    config: Mapping[str, Any],
    challenge_id: str,
    replicate: int,
    component: str,
) -> np.random.Generator:
    material = (
        f"{config['candidate']['deterministic_seed']}|{challenge_id}|"
        f"{replicate}|{component}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(np.random.SeedSequence(list(digest)))


def _peak_draw_matrix(
    successes: np.ndarray,
    trials: np.ndarray,
    *,
    draws: int,
    prior_alpha: float,
    prior_beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    alpha = successes + prior_alpha
    beta = trials - successes + prior_beta
    probability_draws = rng.beta(
        alpha[np.newaxis, :, :],
        beta[np.newaxis, :, :],
        size=(draws, successes.shape[0], successes.shape[1]),
    )
    return np.argmax(probability_draws, axis=2).astype(np.int64)


def _peak_set_diagnostic(
    peak_indices: np.ndarray,
    grid: np.ndarray,
    *,
    credible_mass: float,
    minimum_modal_probability: float,
    maximum_span: int,
    require_contiguous: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    counts = np.bincount(peak_indices, minlength=grid.size)
    probabilities = counts / counts.sum()
    mode_index = int(np.argmax(probabilities))
    order = np.argsort(-probabilities, kind="stable")
    cumulative = 0.0
    selected: list[int] = []
    for index in order:
        selected.append(int(index))
        cumulative += float(probabilities[index])
        if cumulative >= credible_mass:
            break
    selected.sort()
    reasons: list[str] = []
    if mode_index in (0, grid.size - 1):
        reasons.append("BOUNDARY_MODAL_PEAK")
    if float(probabilities[mode_index]) < minimum_modal_probability:
        reasons.append("DIFFUSE_PEAK_PROBABILITY")
    if selected[-1] - selected[0] > maximum_span:
        reasons.append("WIDE_PEAK_SET")
    if require_contiguous and selected != list(range(selected[0], selected[-1] + 1)):
        reasons.append("NONCONTIGUOUS_PEAK_SET")
    return (
        not reasons,
        reasons,
        {
            "modal_peak": float(grid[mode_index]),
            "modal_peak_probability": float(probabilities[mode_index]),
            "peak_set": [float(grid[index]) for index in selected],
            "peak_probabilities": probabilities.tolist(),
        },
    )


def _estimate_panel(
    config: Mapping[str, Any],
    panel: Mapping[str, Any],
    posterior_draws: int,
) -> dict[str, Any]:
    require(set(panel) == PANEL_KEYS, "panel exposes unexpected fields")
    challenge_id = str(panel["challenge_id"])
    replicate = int(panel["replicate"])
    grid = np.asarray(panel["grid"], dtype=np.float64)
    require(tuple(grid.tolist()) == EXPECTED_GRID, "panel grid drift")
    episodes = panel["episodes"]
    require(isinstance(episodes, list) and episodes, "episodes missing")
    for episode in episodes:
        require(set(episode) == EPISODE_KEYS, "episode exposes unexpected fields")
    opportunities = np.asarray(
        [episode["opportunities"] for episode in episodes], dtype=np.int64
    )
    entries = np.asarray([episode["entries"] for episode in episodes], dtype=np.int64)
    persistent = np.asarray(
        [episode["persistent"] for episode in episodes], dtype=np.int64
    )
    require(
        opportunities.shape == entries.shape == persistent.shape == (len(episodes), grid.size),
        "panel array shape drift",
    )
    require(np.all((0 <= persistent) & (persistent <= entries)), "persistence count drift")
    require(np.all((0 <= entries) & (entries <= opportunities)), "entry count drift")
    candidate = config["candidate"]
    prior = candidate["binomial_rate_model"]
    posterior = candidate["posterior_peak"]
    entry_peaks = _peak_draw_matrix(
        entries,
        opportunities,
        draws=posterior_draws,
        prior_alpha=float(prior["prior_alpha"]),
        prior_beta=float(prior["prior_beta"]),
        rng=_rng(config, challenge_id, replicate, "entry"),
    )
    persistence_peaks = _peak_draw_matrix(
        persistent,
        entries,
        draws=posterior_draws,
        prior_alpha=float(prior["prior_alpha"]),
        prior_beta=float(prior["prior_beta"]),
        rng=_rng(config, challenge_id, replicate, "persistence"),
    )
    gap_draws = grid[entry_peaks] - grid[persistence_peaks]
    invalid_reason_counts: Counter[str] = Counter()
    valid_episode_positions: dict[int, list[int]] = defaultdict(list)
    valid_gap_draws: dict[int, np.ndarray] = {}
    episode_diagnostics: list[dict[str, Any]] = []
    for position, episode in enumerate(episodes):
        reasons: list[str] = []
        if np.any(opportunities[position] <= 0) or np.any(entries[position] <= 0):
            reasons.append("MISSING_REGISTERED_TIME")
        entry_valid, entry_reasons, entry_diagnostic = _peak_set_diagnostic(
            entry_peaks[:, position],
            grid,
            credible_mass=float(posterior["peak_set_credible_mass"]),
            minimum_modal_probability=float(
                posterior["minimum_modal_peak_probability"]
            ),
            maximum_span=int(posterior["maximum_peak_set_grid_span"]),
            require_contiguous=bool(posterior["require_contiguous_peak_set"]),
        )
        persistence_valid, persistence_reasons, persistence_diagnostic = (
            _peak_set_diagnostic(
                persistence_peaks[:, position],
                grid,
                credible_mass=float(posterior["peak_set_credible_mass"]),
                minimum_modal_probability=float(
                    posterior["minimum_modal_peak_probability"]
                ),
                maximum_span=int(posterior["maximum_peak_set_grid_span"]),
                require_contiguous=bool(posterior["require_contiguous_peak_set"]),
            )
        )
        reasons.extend(entry_reasons)
        reasons.extend(persistence_reasons)
        reasons = sorted(set(reasons))
        for reason in reasons:
            invalid_reason_counts[reason] += 1
        valid = not reasons and entry_valid and persistence_valid
        field_id = int(episode["field_id"])
        if valid:
            valid_episode_positions[field_id].append(position)
            valid_gap_draws[position] = gap_draws[:, position]
        episode_diagnostics.append(
            {
                "field_id": field_id,
                "episode_id": int(episode["episode_id"]),
                "valid": valid,
                "invalid_reasons": reasons,
                "entry": entry_diagnostic,
                "persistence": persistence_diagnostic,
            }
        )
    aggregation = candidate["aggregation"]
    field_count = int(aggregation["field_count"])
    minimum = int(aggregation["minimum_valid_episodes_per_field"])
    panel_valid = len(valid_episode_positions) == field_count and all(
        len(valid_episode_positions[field_id]) >= minimum
        for field_id in range(field_count)
    )
    supported = sum(map(len, valid_episode_positions.values()))
    if not panel_valid:
        invalid_reason_counts["INSUFFICIENT_VALID_EPISODES_PER_FIELD"] += 1
        return {
            "challenge_id": challenge_id,
            "replicate": replicate,
            "candidate": CANDIDATE_NAME,
            "valid": False,
            "point_estimate": None,
            "one_sided_lower": None,
            "two_sided_lower": None,
            "two_sided_upper": None,
            "passed": False,
            "valid_episodes": supported,
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "episode_diagnostics": episode_diagnostics,
        }
    point = float(
        np.mean(
            [
                np.mean(
                    [
                        np.mean(valid_gap_draws[position])
                        for position in valid_episode_positions[field_id]
                    ]
                )
                for field_id in range(field_count)
            ]
        )
    )
    combined_rng = _rng(config, challenge_id, replicate, "field_episode_resampling")
    combined = np.empty(posterior_draws, dtype=np.float64)
    draw_indices = np.arange(posterior_draws)
    for draw_index in range(posterior_draws):
        field_means: list[float] = []
        for field_id in range(field_count):
            positions = np.asarray(valid_episode_positions[field_id], dtype=np.int64)
            sampled = combined_rng.choice(positions, size=positions.size, replace=True)
            values = np.asarray(
                [valid_gap_draws[int(position)][draw_indices[draw_index]] for position in sampled],
                dtype=np.float64,
            )
            field_means.append(float(np.mean(values)))
        combined[draw_index] = float(np.mean(field_means))
    one_sided_lower = float(np.quantile(combined, 0.05))
    two_sided_lower = float(np.quantile(combined, 0.025))
    two_sided_upper = float(np.quantile(combined, 0.975))
    decision = candidate["decision_rule"]
    passed = bool(
        point >= float(decision["minimum_point_gap_cohort_years"])
        and one_sided_lower
        > float(decision["one_sided_lower_bound_strictly_greater_than"])
    )
    return {
        "challenge_id": challenge_id,
        "replicate": replicate,
        "candidate": CANDIDATE_NAME,
        "valid": True,
        "point_estimate": point,
        "one_sided_lower": one_sided_lower,
        "two_sided_lower": two_sided_lower,
        "two_sided_upper": two_sided_upper,
        "passed": passed,
        "valid_episodes": supported,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "episode_diagnostics": episode_diagnostics,
    }


def plan(
    config_path: Path,
    *,
    source_binding_scope: str = "panels",
) -> dict[str, Any]:
    config = load_config(
        config_path,
        source_binding_scope=source_binding_scope,
    )
    return {
        "status": "PASS_PEAK_SET_DEVELOPMENT_PLAN_NO_ESTIMATION_NO_DIAGNOSTIC",
        "config_sha256": _sha256(config_path),
        "candidate": CANDIDATE_NAME,
        "profiles": config["candidate"]["profiles"],
        "source_role": config["source_diagnostic"]["role"],
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "scientific_effect": "NOT_ESTABLISHED",
    }


def estimate(
    config_path: Path,
    output_root: Path,
    profile_name: str,
) -> dict[str, Any]:
    config = load_config(config_path, source_binding_scope="panels")
    profiles = config["candidate"]["profiles"]
    require(profile_name in profiles, f"unknown profile: {profile_name}")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=False)
    require(stat.S_IMODE(output_root.stat().st_mode) == 0o700, "output mode drift")
    source_root = Path(config["source_diagnostic"]["root"])
    panels_path = source_root / "challenge_panels.jsonl"
    panels = _read_jsonl(panels_path)
    posterior_draws = int(profiles[profile_name]["posterior_draws"])
    rows: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(panels, start=1):
        rows.append(_estimate_panel(config, panel, posterior_draws))
        if panel_index % 512 == 0 or panel_index == len(panels):
            print(
                json.dumps(
                    {
                        "status": "RUNNING_OPENED_SYNTHETIC_DISCOVERY_ONLY",
                        "completed_panels": panel_index,
                        "total_panels": len(panels),
                        "valid_panels": sum(bool(row["valid"]) for row in rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    estimates_path = output_root / "candidate_estimates.jsonl"
    _write_jsonl(estimates_path, rows)
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_peak_set_estimation_receipt.v1",
        "status": "PASS_OPENED_SYNTHETIC_CANDIDATE_ESTIMATES_NO_PRIVATE_LABEL_READ",
        "completed_at_utc": _utc_now(),
        "profile": profile_name,
        "config_sha256": _sha256(config_path),
        "developer_sha256": _sha256(Path(__file__).resolve()),
        "source_panels_sha256": _sha256(panels_path),
        "candidate_estimates": {
            "rows": len(rows),
            "bytes": estimates_path.stat().st_size,
            "sha256": _sha256(estimates_path),
        },
        "opened_private_label_files_read": 0,
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "access_accounting": {
            "source_or_empirical_rows_read": 0,
            "current_g1_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "models_fitted_to_empirical_data": 0,
        },
        "scientific_effect": "NOT_ESTABLISHED",
    }
    receipt_path = output_root / "estimation_receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output_root": str(output_root),
        "profile": profile_name,
        "panel_rows": len(rows),
        "valid_panels": sum(bool(row["valid"]) for row in rows),
        "candidate_estimates_sha256": _sha256(estimates_path),
        "estimation_receipt_sha256": _sha256(receipt_path),
        "opened_private_label_files_read": 0,
    }


def _diagnostic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identifiable = [row for row in rows if bool(row["truth_identifiable"])]
    nonidentifiable = [row for row in rows if not bool(row["truth_identifiable"])]
    identifiable_valid = [row for row in identifiable if bool(row["valid"])]
    nonidentifiable_valid = [row for row in nonidentifiable if bool(row["valid"])]
    meaningful = [
        row
        for row in identifiable_valid
        if float(row["truth_delta"]) >= 1.0
    ]
    nonpositive = [
        row for row in identifiable_valid if float(row["truth_delta"]) <= 0.0
    ]
    signed = [
        row
        for row in identifiable_valid
        if not np.isclose(float(row["truth_delta"]), 0.0)
    ]
    coverage = [
        float(row["two_sided_lower"])
        <= float(row["truth_delta"])
        <= float(row["two_sided_upper"])
        for row in identifiable_valid
    ]
    reversals = [
        (float(row["truth_delta"]) > 0.0 and float(row["point_estimate"]) < 0.0)
        or (float(row["truth_delta"]) < 0.0 and float(row["point_estimate"]) > 0.0)
        for row in signed
    ]
    no_feedback_valid = [
        row for row in rows if bool(row["valid"]) and not bool(row["feedback_present"])
    ]
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(row["invalid_reason_counts"])

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "rows": len(rows),
        "identifiable_valid_rate": rate(len(identifiable_valid), len(identifiable)),
        "nonidentifiable_invalid_rate": rate(
            len(nonidentifiable) - len(nonidentifiable_valid), len(nonidentifiable)
        ),
        "meaningful_positive_detection_rate_among_valid": rate(
            sum(bool(row["passed"]) for row in meaningful), len(meaningful)
        ),
        "nonpositive_false_positive_rate_among_valid": rate(
            sum(bool(row["passed"]) for row in nonpositive), len(nonpositive)
        ),
        "identifiable_two_sided_coverage_among_valid": rate(
            sum(coverage), len(coverage)
        ),
        "identifiable_sign_reversal_rate_among_valid": rate(
            sum(reversals), len(reversals)
        ),
        "mechanism_overreach_pass_rate_among_valid": rate(
            sum(bool(row["passed"]) for row in no_feedback_valid),
            len(no_feedback_valid),
        ),
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "denominators": {
            "identifiable": len(identifiable),
            "identifiable_valid": len(identifiable_valid),
            "nonidentifiable": len(nonidentifiable),
            "nonidentifiable_valid": len(nonidentifiable_valid),
            "meaningful_positive_valid": len(meaningful),
            "nonpositive_valid": len(nonpositive),
            "identifiable_signed_valid": len(signed),
            "no_feedback_valid": len(no_feedback_valid),
        },
    }


def diagnose(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    estimates_path = output_root / "candidate_estimates.jsonl"
    estimation_receipt_path = output_root / "estimation_receipt.json"
    report_path = output_root / "diagnostic_report.json"
    receipt_path = output_root / "diagnostic_receipt.json"
    require(not report_path.exists(), "diagnostic report already exists")
    require(not receipt_path.exists(), "diagnostic receipt already exists")
    estimation = _read_json(estimation_receipt_path)
    require(
        estimation.get("status")
        == "PASS_OPENED_SYNTHETIC_CANDIDATE_ESTIMATES_NO_PRIVATE_LABEL_READ",
        "candidate estimation incomplete",
    )
    require(estimation.get("config_sha256") == _sha256(config_path), "config drift")
    require(
        estimation["candidate_estimates"].get("sha256") == _sha256(estimates_path),
        "candidate estimate binding drift",
    )
    source_root = Path(config["source_diagnostic"]["root"])
    labels_payload = _read_json(source_root / "private_labels.json")
    labels = labels_payload.get("labels")
    require(isinstance(labels, list) and labels, "opened labels missing")
    label_map = {
        (str(row["challenge_id"]), int(row["replicate"])): row for row in labels
    }
    estimates = _read_jsonl(estimates_path)
    require(len(estimates) == len(label_map), "estimate/label cardinality drift")
    joined: list[dict[str, Any]] = []
    for estimate in estimates:
        key = (str(estimate["challenge_id"]), int(estimate["replicate"]))
        require(key in label_map, f"missing label: {key}")
        joined.append({**estimate, **label_map[key]})
    overall = _diagnostic_metrics(joined)
    by_family = {
        family: _diagnostic_metrics(
            [row for row in joined if row["generator_family"] == family]
        )
        for family in sorted({str(row["generator_family"]) for row in joined})
    }
    frozen_report = _read_json(source_root / "unblinded_report.json")
    report = {
        "schema_version": "pivot_feedback_lean_h3_peak_set_diagnostic_report.v1",
        "status": "PASS_OPENED_SYNTHETIC_DISCOVERY_DIAGNOSTIC_NO_ADMISSIBILITY",
        "completed_at_utc": _utc_now(),
        "candidate": CANDIDATE_NAME,
        "config_sha256": _sha256(config_path),
        "source_role": config["source_diagnostic"]["role"],
        "metrics": overall,
        "family_metrics": by_family,
        "frozen_v1_reference": {
            "role": "CONTEXT_ONLY_NOT_REJUDGED",
            "status": frozen_report["status"],
            "estimator_metrics": frozen_report["estimator_metrics"],
        },
        "disposition": {
            "candidate_admissibility": "NOT_EVALUATED",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "opened_challenge_reuse_as_blind_confirmation": "FORBIDDEN",
            "fresh_blind_promotion": "REQUIRED_SEPARATE_PRECOMMIT_AND_DISJOINT_GENERATOR",
        },
        "access_accounting": estimation["access_accounting"],
        "authorization": {
            "opened_synthetic_private_label_read": "COMPLETE_DIAGNOSTIC_ONLY",
            "empirical_row_access": "FORBIDDEN_NOT_RUN",
            "confirmatory_execution": "FORBIDDEN_NOT_RUN",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(report_path, report)
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_peak_set_diagnostic_receipt.v1",
        "status": "PASS_DISCOVERY_DIAGNOSTIC_RECEIPT_WRITTEN_LAST_NO_SELECTION",
        "completed_at_utc": _utc_now(),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "developer_sha256": _sha256(Path(__file__).resolve()),
            "estimation_receipt_sha256": _sha256(estimation_receipt_path),
            "candidate_estimates_sha256": _sha256(estimates_path),
            "opened_private_labels_sha256": _sha256(
                source_root / "private_labels.json"
            ),
            "diagnostic_report_sha256": _sha256(report_path),
        },
        "candidate_admissibility": "NOT_EVALUATED",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "access_accounting": estimation["access_accounting"],
        "scientific_effect": "NOT_ESTABLISHED",
    }
    _write_json(receipt_path, receipt)
    return {
        "status": report["status"],
        "candidate_admissibility": "NOT_EVALUATED",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "diagnostic_report_sha256": _sha256(report_path),
        "diagnostic_receipt_sha256": _sha256(receipt_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "estimate", "diagnose"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", default="fixture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        require(args.output_root is None, "plan forbids --output-root")
        result = plan(args.config)
    else:
        require(args.output_root is not None, f"{args.command} requires --output-root")
        result = (
            estimate(args.config, args.output_root, args.profile)
            if args.command == "estimate"
            else diagnose(args.config, args.output_root)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
