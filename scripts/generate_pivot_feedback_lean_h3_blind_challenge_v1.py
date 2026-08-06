#!/usr/bin/env python3
"""Generate opaque source-independent panels for the lean-H3 blind challenge."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_lean_h3_blind_challenge.v1"
EXPECTED_MODE = (
    "SYNTHETIC_ONLY_GENERATOR_HELD_OUT_PROCEDURAL_BLIND_ESTIMATOR_CHALLENGE"
)
EXPECTED_STATUS = "PRECOMMITTED_NO_CHALLENGE_SEED_NO_PANEL_NO_RESULT"
EXPECTED_GRID = (-2, -1, 0, 1, 2)
EXPECTED_FAMILIES = (
    "asymmetric_peak_and_tail",
    "plateau_and_near_tie",
    "staggered_field_mixture",
    "correlated_episode_shift",
    "informative_time_missingness",
    "rounded_heaped_measurement",
    "monotone_boundary_process",
    "no_feedback_observational_mimic",
)
TRACKED_CHALLENGE_PATHS = (
    "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml",
    "scripts/generate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/estimate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/verify_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "tests/test_pivot_feedback_lean_h3_blind_challenge_v1.py",
)


class BlindChallengeError(RuntimeError):
    """Raised when challenge generation cannot preserve its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BlindChallengeError(message)


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


def _load_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "config root must be a mapping")
    return payload


def _positive_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def _finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def load_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("mode") == EXPECTED_MODE, "config mode drift")
    require(config.get("status") == EXPECTED_STATUS, "config status drift")
    scientific = config.get("scientific_object")
    require(isinstance(scientific, dict), "scientific_object missing")
    require(
        tuple(scientific.get("event_time_grid", ())) == EXPECTED_GRID,
        "event-time grid drift",
    )
    require(
        tuple(config.get("held_out_generator_families", ())) == EXPECTED_FAMILIES,
        "held-out family drift",
    )
    design = config.get("design")
    require(isinstance(design, dict), "design missing")
    _positive_int(design.get("field_count"), "field_count")
    episodes = _positive_int(design.get("episodes_per_field"), "episodes_per_field")
    minimum = _positive_int(
        design.get("minimum_supported_episodes_per_field"),
        "minimum_supported_episodes_per_field",
    )
    require(minimum <= episodes, "minimum episodes exceed total episodes")
    _positive_int(
        design.get("minimum_finite_event_times_per_curve"),
        "minimum_finite_event_times_per_curve",
    )
    profiles = design.get("profiles")
    require(isinstance(profiles, dict) and profiles, "profiles missing")
    for name, profile in profiles.items():
        require(isinstance(name, str) and isinstance(profile, dict), "invalid profile")
        for key in (
            "scenarios_per_family",
            "replicates_per_scenario",
            "bootstrap_replicates",
        ):
            _positive_int(profile.get(key), f"{name}.{key}")
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
    require(authorization.get("estimator_selected") == "NO", "estimator selection drift")
    require(authorization.get("support_rule_selected") == "NO", "support selection drift")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")
    for binding in config["source_result_space"].values():
        if not isinstance(binding, dict) or "relative_path" not in binding:
            continue
        bound_path = ROOT / str(binding["relative_path"])
        require(_sha256(bound_path) == binding.get("sha256"), f"source binding drift: {bound_path}")
    return config


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_precommit(commit: str) -> None:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid precommit hash")
    require(_git_head() == commit, "HEAD differs from declared precommit")
    result = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *TRACKED_CHALLENGE_PATHS],
        cwd=ROOT,
        check=False,
    )
    require(result.returncode == 0, "challenge code differs from precommit")


def _rng_from_seed(seed_hex: str) -> np.random.Generator:
    require(bool(re.fullmatch(r"[0-9a-f]{64}", seed_hex)), "seed must be 256-bit hex")
    return np.random.default_rng(np.random.SeedSequence(list(bytes.fromhex(seed_hex))))


def _earliest_peak(grid: np.ndarray, values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return float(grid[np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-12))[0]])


def _gaussian(grid: np.ndarray, peak: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid - peak) / width) ** 2)


def _asymmetric(grid: np.ndarray, peak: float, width: float, skew: float) -> np.ndarray:
    distance = grid - peak
    local_width = np.where(distance < 0.0, width * (1.0 - skew), width * (1.0 + skew))
    return np.exp(-0.5 * (distance / local_width) ** 2)


def _scenario_spec(
    family: str,
    scenario_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    family_index = EXPECTED_FAMILIES.index(family)
    delta = (-1, 0, 1, 2)[(scenario_index + family_index) % 4]
    persistence_peak = int(rng.choice(np.asarray([-1, 0, 1])))
    entry_peak = int(np.clip(persistence_peak + delta, -2, 2))
    delta = entry_peak - persistence_peak
    spec: dict[str, Any] = {
        "internal_id": f"{family}-{scenario_index}",
        "family": family,
        "feedback_present": family in {
            "asymmetric_peak_and_tail",
            "staggered_field_mixture",
            "correlated_episode_shift",
        }
        and delta > 0,
        "truth_identifiable": family
        not in {
            "plateau_and_near_tie",
            "informative_time_missingness",
            "monotone_boundary_process",
        },
        "entry_peak": float(entry_peak),
        "persistence_peak": float(persistence_peak),
        "entry_width": float(rng.uniform(0.45, 1.25)),
        "persistence_width": float(rng.uniform(0.45, 1.25)),
        "skew": float(rng.uniform(0.15, 0.55)),
        "opportunity_level": int(rng.integers(600, 8001)),
        "base_entry_probability": float(rng.uniform(0.003, 0.030)),
        "entry_amplitude": float(rng.uniform(0.025, 0.100)),
        "base_persistence_probability": float(rng.uniform(0.15, 0.45)),
        "persistence_amplitude": float(rng.uniform(0.15, 0.48)),
        "episode_jitter_sd": float(rng.uniform(0.0, 0.38)),
    }
    if family == "no_feedback_observational_mimic":
        spec["feedback_present"] = False
        spec["truth_identifiable"] = True
        spec["persistence_peak"] = float(rng.choice(np.asarray([-1, 0])))
        spec["entry_peak"] = spec["persistence_peak"] + float(rng.choice(np.asarray([1, 2])))
    if family == "plateau_and_near_tie":
        spec["entry_peak"] = float(rng.choice(np.asarray([-1, 0])))
        spec["persistence_peak"] = float(rng.choice(np.asarray([-1, 0])))
    if family == "monotone_boundary_process":
        spec["feedback_present"] = False
    return spec


def _probability_curves(
    grid: np.ndarray,
    spec: Mapping[str, Any],
    *,
    field_id: int,
    episode_id: int,
    field_shift: float,
    episode_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    family = str(spec["family"])
    entry_peak = float(spec["entry_peak"]) + episode_shift
    persistence_peak = float(spec["persistence_peak"])
    if family == "staggered_field_mixture":
        entry_peak += (-1.25, -0.25, 0.75, 1.25)[field_id]
        persistence_peak += (0.75, -0.75, 0.25, -0.25)[field_id]
    if family == "correlated_episode_shift":
        entry_peak += field_shift
        persistence_peak -= 0.5 * field_shift

    if family == "asymmetric_peak_and_tail":
        entry_shape = _asymmetric(
            grid,
            entry_peak,
            float(spec["entry_width"]),
            float(spec["skew"]),
        )
        persistence_shape = _asymmetric(
            grid,
            persistence_peak,
            float(spec["persistence_width"]),
            -float(spec["skew"]),
        )
    elif family == "plateau_and_near_tie":
        entry_shape = _gaussian(grid, entry_peak, 0.52)
        persistence_shape = _gaussian(grid, persistence_peak, 0.52)
        entry_positions = np.argsort(np.abs(grid - entry_peak))[:2]
        persistence_positions = np.argsort(np.abs(grid - persistence_peak))[:2]
        entry_shape[entry_positions] = 1.0
        persistence_shape[persistence_positions] = 1.0
    elif family == "monotone_boundary_process":
        direction = 1.0 if episode_id % 2 == 0 else -1.0
        entry_shape = np.linspace(0.05, 1.0, grid.size)
        persistence_shape = np.linspace(1.0, 0.05, grid.size)
        if direction < 0:
            entry_shape = entry_shape[::-1]
            persistence_shape = persistence_shape[::-1]
    else:
        entry_shape = _gaussian(grid, entry_peak, float(spec["entry_width"]))
        persistence_shape = _gaussian(
            grid,
            persistence_peak,
            float(spec["persistence_width"]),
        )

    entry_probability = np.clip(
        float(spec["base_entry_probability"])
        + float(spec["entry_amplitude"]) * entry_shape,
        0.0001,
        0.35,
    )
    persistence_probability = np.clip(
        float(spec["base_persistence_probability"])
        + float(spec["persistence_amplitude"]) * persistence_shape,
        0.01,
        0.95,
    )
    if family == "rounded_heaped_measurement":
        entry_probability = np.round(entry_probability / 0.01) * 0.01
        persistence_probability = np.round(persistence_probability / 0.05) * 0.05
    return entry_probability, persistence_probability


def _panel(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    challenge_id: str,
    replicate: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grid = np.asarray(EXPECTED_GRID, dtype=np.float64)
    design = config["design"]
    field_count = int(design["field_count"])
    episodes_per_field = int(design["episodes_per_field"])
    episodes: list[dict[str, Any]] = []
    truth_by_field: dict[int, list[float]] = {field_id: [] for field_id in range(field_count)}
    family = str(spec["family"])
    field_shifts = rng.normal(0.0, 0.55, size=field_count)
    for field_id in range(field_count):
        for episode_id in range(episodes_per_field):
            episode_shift = float(rng.normal(0.0, float(spec["episode_jitter_sd"])))
            entry_probability, persistence_probability = _probability_curves(
                grid,
                spec,
                field_id=field_id,
                episode_id=episode_id,
                field_shift=float(field_shifts[field_id]),
                episode_shift=episode_shift,
            )
            truth_by_field[field_id].append(
                _earliest_peak(grid, entry_probability)
                - _earliest_peak(grid, persistence_probability)
            )
            opportunities = rng.poisson(
                float(spec["opportunity_level"]), size=grid.size
            ).astype(np.int64)
            if family == "informative_time_missingness":
                peak_position = int(np.argmax(entry_probability))
                opportunities[peak_position] = 0
                if (field_id + episode_id + replicate) % 3 == 0:
                    opportunities[max(0, peak_position - 1)] = 0
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
    truth_delta = float(
        np.mean([np.mean(truth_by_field[field_id]) for field_id in range(field_count)])
    )
    panel = {
        "challenge_id": challenge_id,
        "replicate": replicate,
        "grid": list(EXPECTED_GRID),
        "episodes": episodes,
    }
    label = {
        "challenge_id": challenge_id,
        "replicate": replicate,
        "generator_family": family,
        "feedback_present": bool(spec["feedback_present"]),
        "truth_identifiable": bool(spec["truth_identifiable"]),
        "truth_delta": truth_delta,
    }
    return panel, label


def plan(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "status": "PASS_BLIND_CHALLENGE_PLAN_NO_SEED_NO_PANEL_NO_RESULT",
        "config_sha256": _sha256(config_path),
        "profiles": config["design"]["profiles"],
        "held_out_generator_family_count": len(EXPECTED_FAMILIES),
        "estimators": config["design"]["estimators"],
        "source_or_empirical_row_read": "FORBIDDEN",
        "scientific_effect": "NOT_ESTABLISHED",
    }


def generate(
    config_path: Path,
    output_root: Path,
    profile_name: str,
    *,
    precommit: str,
    seed_hex: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    _require_precommit(precommit)
    profiles = config["design"]["profiles"]
    require(profile_name in profiles, f"unknown profile: {profile_name}")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(mode=0o700, parents=False)
    require(stat.S_IMODE(output_root.stat().st_mode) == 0o700, "output mode drift")
    profile = profiles[profile_name]
    seed = seed_hex if seed_hex is not None else secrets.token_hex(32)
    rng = _rng_from_seed(seed)
    specs = [
        _scenario_spec(family, scenario_index, rng)
        for family in EXPECTED_FAMILIES
        for scenario_index in range(int(profile["scenarios_per_family"]))
    ]
    order = rng.permutation(len(specs))
    ordered_specs = [specs[int(index)] for index in order]
    panels: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for opaque_index, spec in enumerate(ordered_specs):
        challenge_id = f"challenge_{opaque_index:04d}"
        for replicate in range(int(profile["replicates_per_scenario"])):
            panel, label = _panel(
                config,
                spec,
                challenge_id,
                replicate,
                rng,
            )
            panels.append(panel)
            labels.append(label)

    panels_path = output_root / "challenge_panels.jsonl"
    labels_path = output_root / "private_labels.json"
    _write_jsonl(panels_path, panels)
    _write_json(
        labels_path,
        {
            "schema_version": "pivot_feedback_lean_h3_blind_private_labels.v1",
            "seed_hex": seed,
            "precommit_git_commit": precommit,
            "profile": profile_name,
            "labels": labels,
        },
    )
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_blind_generation_receipt.v1",
        "status": "PASS_OPAQUE_SYNTHETIC_PANELS_PRIVATE_LABELS_NOT_OPENED_BY_ESTIMATOR",
        "completed_at_utc": _utc_now(),
        "precommit_git_commit": precommit,
        "profile": profile_name,
        "config_sha256": _sha256(config_path),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "challenge_panels": {
            "rows": len(panels),
            "bytes": panels_path.stat().st_size,
            "sha256": _sha256(panels_path),
        },
        "private_label_commitment_sha256": _sha256(labels_path),
        "private_label_access_by_estimator": "FORBIDDEN",
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
    receipt_path = output_root / "generation_receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output_root": str(output_root),
        "profile": profile_name,
        "panel_rows": len(panels),
        "challenge_scenarios": len(ordered_specs),
        "challenge_panels_sha256": _sha256(panels_path),
        "generation_receipt_sha256": _sha256(receipt_path),
        "seed_disclosed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "generate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--precommit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        require(args.output_root is None, "plan forbids --output-root")
        require(args.precommit is None, "plan forbids --precommit")
        result = plan(args.config)
    else:
        require(args.output_root is not None, "generate requires --output-root")
        require(args.precommit is not None, "generate requires --precommit")
        result = generate(
            args.config,
            args.output_root,
            args.profile,
            precommit=args.precommit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
