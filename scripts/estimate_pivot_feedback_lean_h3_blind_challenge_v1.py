#!/usr/bin/env python3
"""Estimate lean-H3 challenge panels without reading their private labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
EXPECTED_ESTIMATORS = ("raw_earliest_argmax", "kernel_earliest_argmax")
EXPECTED_GRID = (-2, -1, 0, 1, 2)
PANEL_KEYS = {"challenge_id", "replicate", "grid", "episodes"}
EPISODE_KEYS = {
    "field_id",
    "episode_id",
    "opportunities",
    "entries",
    "persistent",
}
TRACKED_CHALLENGE_PATHS = (
    "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml",
    "scripts/generate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/estimate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/verify_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "tests/test_pivot_feedback_lean_h3_blind_challenge_v1.py",
)


class BlindEstimateError(RuntimeError):
    """Raised when blinded estimation cannot preserve the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BlindEstimateError(message)


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
            require(isinstance(payload, dict), f"invalid row {line_number}")
            rows.append(payload)
    require(rows, f"empty JSONL input: {path}")
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    design = config.get("design")
    require(isinstance(design, dict), "design missing")
    require(tuple(design.get("estimators", ())) == EXPECTED_ESTIMATORS, "estimator drift")
    require(
        tuple(config["scientific_object"].get("event_time_grid", ())) == EXPECTED_GRID,
        "event-time grid drift",
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
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")
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
    require(_git_head() == commit, "HEAD differs from generation precommit")
    result = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *TRACKED_CHALLENGE_PATHS],
        cwd=ROOT,
        check=False,
    )
    require(result.returncode == 0, "challenge code differs from precommit")


def _kernel_smooth(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.size):
        left = max(0, index - 1)
        right = min(values.size, index + 2)
        positions = np.arange(left, right)
        local_weights = weights[positions - index + 1]
        finite = np.isfinite(values[positions])
        if not np.any(finite):
            continue
        local_weights = local_weights[finite]
        local_weights = local_weights / local_weights.sum()
        smoothed[index] = float(np.dot(values[positions][finite], local_weights))
    return smoothed


def _episode_delta(
    grid: np.ndarray,
    episode: Mapping[str, Any],
    *,
    estimator: str,
    kernel_weights: np.ndarray,
    minimum_finite: int,
) -> float | None:
    require(set(episode) == EPISODE_KEYS, "episode exposes non-blind fields")
    opportunities = np.asarray(episode["opportunities"], dtype=np.float64)
    entries = np.asarray(episode["entries"], dtype=np.float64)
    persistent = np.asarray(episode["persistent"], dtype=np.float64)
    require(
        opportunities.shape == entries.shape == persistent.shape == grid.shape,
        "curve length drift",
    )
    entry_rate = np.divide(
        entries,
        opportunities,
        out=np.full_like(entries, np.nan),
        where=opportunities > 0,
    )
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

    def earliest(values: np.ndarray) -> float:
        indices = np.flatnonzero(np.isfinite(values))
        local = values[indices]
        maximum = float(np.max(local))
        local_index = np.flatnonzero(
            np.isclose(local, maximum, rtol=0.0, atol=1e-12)
        )[0]
        return float(grid[int(indices[int(local_index)])])

    return earliest(entry_rate) - earliest(persistence_rate)


def _field_balanced(values: Mapping[int, Sequence[float]]) -> float:
    return float(np.mean([np.mean(field_values) for field_values in values.values()]))


def _bootstrap_rng(
    config: Mapping[str, Any],
    challenge_id: str,
    replicate: int,
    estimator: str,
) -> np.random.Generator:
    material = (
        f"{config['design']['bootstrap_seed']}|{challenge_id}|{replicate}|{estimator}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(np.random.SeedSequence(list(digest)))


def _estimate_panel(
    config: Mapping[str, Any],
    panel: Mapping[str, Any],
    estimator: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    require(set(panel) == PANEL_KEYS, "panel exposes non-blind fields")
    challenge_id = panel.get("challenge_id")
    replicate = panel.get("replicate")
    require(isinstance(challenge_id, str) and challenge_id, "challenge_id missing")
    require(isinstance(replicate, int) and replicate >= 0, "invalid replicate")
    grid = np.asarray(panel["grid"], dtype=np.float64)
    require(tuple(grid.tolist()) == EXPECTED_GRID, "panel grid drift")
    design = config["design"]
    weights = np.asarray(design["kernel_weights"], dtype=np.float64)
    minimum_finite = int(design["minimum_finite_event_times_per_curve"])
    minimum_episodes = int(design["minimum_supported_episodes_per_field"])
    field_count = int(design["field_count"])
    by_field: dict[int, list[float]] = defaultdict(list)
    for episode in panel["episodes"]:
        value = _episode_delta(
            grid,
            episode,
            estimator=estimator,
            kernel_weights=weights,
            minimum_finite=minimum_finite,
        )
        if value is not None:
            by_field[int(episode["field_id"])].append(value)
    valid = len(by_field) == field_count and all(
        len(by_field[field_id]) >= minimum_episodes for field_id in range(field_count)
    )
    supported = sum(map(len, by_field.values()))
    if not valid:
        return {
            "valid": False,
            "point_estimate": None,
            "one_sided_lower": None,
            "two_sided_lower": None,
            "two_sided_upper": None,
            "passed": False,
            "supported_episodes": supported,
            "episode_nonpositive_share": None,
        }
    point = _field_balanced(by_field)
    rng = _bootstrap_rng(config, challenge_id, replicate, estimator)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for bootstrap_index in range(bootstrap_replicates):
        sampled: dict[int, list[float]] = {}
        for field_id in range(field_count):
            values = np.asarray(by_field[field_id], dtype=np.float64)
            indices = rng.integers(0, values.size, size=values.size)
            sampled[field_id] = values[indices].tolist()
        bootstrap[bootstrap_index] = _field_balanced(sampled)
    one_sided_lower = float(np.quantile(bootstrap, 0.05))
    two_sided_lower = float(np.quantile(bootstrap, 0.025))
    two_sided_upper = float(np.quantile(bootstrap, 0.975))
    decision = design["decision_rule"]
    passed = bool(
        point >= float(decision["minimum_point_gap_cohort_years"])
        and one_sided_lower
        > float(decision["one_sided_lower_bound_strictly_greater_than"])
    )
    values = np.asarray(
        [value for field_values in by_field.values() for value in field_values],
        dtype=np.float64,
    )
    return {
        "valid": True,
        "point_estimate": point,
        "one_sided_lower": one_sided_lower,
        "two_sided_lower": two_sided_lower,
        "two_sided_upper": two_sided_upper,
        "passed": passed,
        "supported_episodes": supported,
        "episode_nonpositive_share": float(np.mean(values <= 0.0)),
    }


def estimate(config_path: Path, challenge_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    panels_path = challenge_root / "challenge_panels.jsonl"
    generation_receipt_path = challenge_root / "generation_receipt.json"
    estimates_path = challenge_root / "blind_estimates.jsonl"
    receipt_path = challenge_root / "blind_estimation_receipt.json"
    require(not estimates_path.exists(), "blind estimates already exist")
    require(not receipt_path.exists(), "blind estimation receipt already exists")
    generation = _read_json(generation_receipt_path)
    require(
        generation.get("status")
        == "PASS_OPAQUE_SYNTHETIC_PANELS_PRIVATE_LABELS_NOT_OPENED_BY_ESTIMATOR",
        "generation receipt status drift",
    )
    require(generation.get("config_sha256") == _sha256(config_path), "config binding drift")
    require(
        generation["challenge_panels"].get("sha256") == _sha256(panels_path),
        "panel binding drift",
    )
    precommit = str(generation.get("precommit_git_commit"))
    _require_precommit(precommit)
    profile_name = str(generation.get("profile"))
    require(profile_name in config["design"]["profiles"], "profile drift")
    bootstrap_replicates = int(
        config["design"]["profiles"][profile_name]["bootstrap_replicates"]
    )
    panels = _read_jsonl(panels_path)
    rows: list[dict[str, Any]] = []
    for panel in panels:
        for estimator in EXPECTED_ESTIMATORS:
            rows.append(
                {
                    "challenge_id": panel["challenge_id"],
                    "replicate": panel["replicate"],
                    "estimator": estimator,
                    **_estimate_panel(
                        config,
                        panel,
                        estimator,
                        bootstrap_replicates,
                    ),
                }
            )
    _write_jsonl(estimates_path, rows)
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_blind_estimation_receipt.v1",
        "status": "PASS_BLIND_ESTIMATES_WRITTEN_BEFORE_UNBLINDING",
        "completed_at_utc": _utc_now(),
        "precommit_git_commit": precommit,
        "profile": profile_name,
        "config_sha256": _sha256(config_path),
        "estimator_sha256": _sha256(Path(__file__).resolve()),
        "generation_receipt_sha256": _sha256(generation_receipt_path),
        "challenge_panels_sha256": _sha256(panels_path),
        "blind_estimates": {
            "rows": len(rows),
            "bytes": estimates_path.stat().st_size,
            "sha256": _sha256(estimates_path),
        },
        "private_label_files_opened": 0,
        "access_accounting": generation["access_accounting"],
        "scientific_effect": "NOT_ESTABLISHED",
    }
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "profile": profile_name,
        "panel_rows": len(panels),
        "estimate_rows": len(rows),
        "blind_estimates_sha256": _sha256(estimates_path),
        "blind_estimation_receipt_sha256": _sha256(receipt_path),
        "private_label_files_opened": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--challenge-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = estimate(args.config, args.challenge_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
