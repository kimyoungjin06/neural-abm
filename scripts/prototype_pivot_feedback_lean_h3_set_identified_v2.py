#!/usr/bin/env python3
"""Prototype set-valued lean H3 semantics on closed analytic fixtures only."""

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
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_set_identified_prospectus_v2.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_lean_h3_set_identified_prospectus.v2"
RESULT_SCHEMA = "pivot_feedback_lean_h3_set_identified_fixture_result.v2"
EXPECTED_MODE = "ANALYTIC_FIXTURE_ONLY_PROSPECTIVE_DEFINITION_NO_OPENED_PANEL_REPLAY"
EXPECTED_STATUS = (
    "AUTHORIZED_PROSPECTUS_ONLY_NO_CANDIDATE_SELECTION_NO_BLIND_PROMOTION"
)
EXPECTED_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0)
CANDIDATE_NAME = "practical_equivalence_set_field_vector_v2"


class SetIdentifiedProspectusError(RuntimeError):
    """Raised when analytic prospectus boundaries or invariants drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SetIdentifiedProspectusError(message)


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("mode") == EXPECTED_MODE, "config mode drift")
    require(config.get("status") == EXPECTED_STATUS, "config status drift")
    anchors = _mapping(config.get("source_anchors"), "source anchors")
    for name, raw_anchor in anchors.items():
        anchor = _mapping(raw_anchor, f"anchor {name}")
        relative_path = Path(str(anchor.get("relative_path")))
        require(not relative_path.is_absolute(), f"absolute anchor path: {name}")
        require(
            _sha256(ROOT / relative_path) == anchor.get("sha256"),
            f"anchor binding drift: {name}",
        )
    scientific = _mapping(config.get("scientific_object"), "scientific object")
    require(
        tuple(float(value) for value in scientific.get("event_time_grid", ()))
        == EXPECTED_GRID,
        "event-time grid drift",
    )
    candidate = _mapping(config.get("candidate_family"), "candidate family")
    require(candidate.get("name") == CANDIDATE_NAME, "candidate name drift")
    curve = _mapping(candidate.get("curve_peak_set"), "curve peak set")
    margin = curve.get("inclusive_absolute_probability_margin")
    require(
        isinstance(margin, (int, float))
        and not isinstance(margin, bool)
        and math.isclose(float(margin), 0.01),
        "practical margin drift",
    )
    require(curve.get("earliest_argmax_tie_break") == "FORBIDDEN", "tie-break drift")
    aggregation = _mapping(candidate.get("aggregation"), "aggregation")
    require(aggregation.get("field_count") == 4, "field count drift")
    require(aggregation.get("episodes_per_field") == 8, "episode count drift")
    require(
        aggregation.get("minimum_valid_episodes_per_field") == 4,
        "minimum episode drift",
    )
    require(aggregation.get("preserve_field_vector") is True, "vector drift")
    require(
        aggregation.get("binary_pass_fail_decision") == "FORBIDDEN",
        "binary decision drift",
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "opened_challenge_panel_replay",
        "opened_challenge_private_label_read",
        "source_or_empirical_row_read",
        "current_g1_row_read",
        "model_selection_row_read",
        "holdout_or_private_g4_read",
        "empirical_parameter_calibration",
        "confirmatory_execution",
    ):
        require(authorization.get(key) == "FORBIDDEN", f"authorization drift: {key}")
    require(authorization.get("candidate_selected") == "NO", "candidate selection drift")
    require(authorization.get("estimator_selected") == "NO", "estimator selection drift")
    require(authorization.get("support_rule_selected") == "NO", "support selection drift")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")
    fixtures = config.get("analytic_fixtures")
    require(isinstance(fixtures, list) and len(fixtures) >= 7, "fixture coverage drift")
    return config


def _curve_peak_set(
    raw_curve: Sequence[Any],
    *,
    margin: float,
) -> dict[str, Any]:
    require(len(raw_curve) == len(EXPECTED_GRID), "curve length drift")
    if any(value is None for value in raw_curve):
        return {
            "valid": False,
            "invalid_reason": "MISSING_REGISTERED_TIME",
            "peak_set": None,
            "shape_state": "INVALID",
        }
    values: list[float] = []
    for value in raw_curve:
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            "curve value must be numeric or null",
        )
        number = float(value)
        require(math.isfinite(number) and 0.0 <= number <= 1.0, "curve value range drift")
        values.append(number)
    maximum = max(values)
    minimum = min(values)
    if maximum - minimum <= margin + 1e-12:
        return {
            "valid": False,
            "invalid_reason": "PRACTICALLY_FLAT_CURVE",
            "peak_set": None,
            "shape_state": "INVALID",
        }
    indices = [
        index
        for index, value in enumerate(values)
        if maximum - value <= margin + 1e-12
    ]
    if 0 in indices or len(EXPECTED_GRID) - 1 in indices:
        return {
            "valid": False,
            "invalid_reason": "BOUNDARY_EQUIVALENT_PEAK_SET",
            "peak_set": None,
            "shape_state": "INVALID",
        }
    contiguous = indices == list(range(indices[0], indices[-1] + 1))
    peak_set = [EXPECTED_GRID[index] for index in indices]
    if len(indices) == 1:
        shape_state = "SINGLETON"
    elif contiguous:
        shape_state = "CONTIGUOUS_SET"
    else:
        shape_state = "MULTIMODAL_SET"
    return {
        "valid": True,
        "invalid_reason": None,
        "peak_set": peak_set,
        "shape_state": shape_state,
    }


def _episode_gap(
    entry: Sequence[Any],
    persistence: Sequence[Any],
    *,
    margin: float,
) -> dict[str, Any]:
    entry_result = _curve_peak_set(entry, margin=margin)
    persistence_result = _curve_peak_set(persistence, margin=margin)
    invalid_reasons = sorted(
        {
            str(result["invalid_reason"])
            for result in (entry_result, persistence_result)
            if not bool(result["valid"])
        }
    )
    if invalid_reasons:
        return {
            "valid": False,
            "invalid_reasons": invalid_reasons,
            "gap_set": None,
            "gap_interval": None,
            "multimodal": False,
        }
    entry_set = [float(value) for value in entry_result["peak_set"]]
    persistence_set = [float(value) for value in persistence_result["peak_set"]]
    gap_set = sorted(
        {entry_value - persistence_value for entry_value in entry_set for persistence_value in persistence_set}
    )
    return {
        "valid": True,
        "invalid_reasons": [],
        "gap_set": gap_set,
        "gap_interval": [min(gap_set), max(gap_set)],
        "multimodal": (
            entry_result["shape_state"] == "MULTIMODAL_SET"
            or persistence_result["shape_state"] == "MULTIMODAL_SET"
        ),
    }


def _fixture_outcome(config: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    candidate = config["candidate_family"]
    margin = float(candidate["curve_peak_set"]["inclusive_absolute_probability_margin"])
    aggregation = candidate["aggregation"]
    repeat = int(aggregation["episodes_per_field"])
    minimum = int(aggregation["minimum_valid_episodes_per_field"])
    field_templates = fixture.get("fields")
    require(isinstance(field_templates, list), "fixture fields missing")
    require(len(field_templates) == int(aggregation["field_count"]), "fixture field count drift")
    field_intervals: list[list[float]] = []
    field_rows: list[dict[str, Any]] = []
    panel_reasons: Counter[str] = Counter()
    any_set = False
    any_multimodal = False
    for expected_field_id, field in enumerate(field_templates):
        require(isinstance(field, dict), "fixture field must be mapping")
        require(field.get("field_id") == expected_field_id, "fixture field order drift")
        episode = _episode_gap(field["entry"], field["persistence"], margin=margin)
        valid_episodes = repeat if episode["valid"] else 0
        if not episode["valid"]:
            panel_reasons.update(episode["invalid_reasons"] * repeat)
        field_valid = valid_episodes >= minimum
        if not field_valid:
            field_rows.append(
                {
                    "field_id": expected_field_id,
                    "valid": False,
                    "valid_episodes": valid_episodes,
                    "interval": None,
                }
            )
            continue
        interval = [float(value) for value in episode["gap_interval"]]
        field_intervals.append(interval)
        any_set = any_set or not math.isclose(interval[0], interval[1])
        any_multimodal = any_multimodal or bool(episode["multimodal"])
        field_rows.append(
            {
                "field_id": expected_field_id,
                "valid": True,
                "valid_episodes": valid_episodes,
                "interval": interval,
            }
        )
    panel_valid = len(field_intervals) == int(aggregation["field_count"])
    if not panel_valid:
        return {
            "fixture_id": fixture["fixture_id"],
            "panel_valid": False,
            "invalid_reasons": sorted(panel_reasons),
            "identification_state": "INVALID",
            "field_intervals": None,
            "field_balanced_interval_hull": None,
            "direction_state": "INVALID_NO_DECISION",
            "field_diagnostics": field_rows,
        }
    lower = sum(interval[0] for interval in field_intervals) / len(field_intervals)
    upper = sum(interval[1] for interval in field_intervals) / len(field_intervals)
    floor = float(
        candidate["descriptive_direction_state"][
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
        identification = "SET_IDENTIFIED_MULTIMODAL_FIELD_VECTOR"
    elif any_set:
        identification = "SET_IDENTIFIED_FIELD_VECTOR"
    else:
        identification = "POINT_IDENTIFIED_FIELD_VECTOR"
    return {
        "fixture_id": fixture["fixture_id"],
        "panel_valid": True,
        "invalid_reasons": [],
        "identification_state": identification,
        "field_intervals": field_intervals,
        "field_balanced_interval_hull": [lower, upper],
        "direction_state": direction,
        "field_diagnostics": field_rows,
    }


def _matches_expected(outcome: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key in (
        "panel_valid",
        "identification_state",
        "field_intervals",
        "field_balanced_interval_hull",
        "direction_state",
    ):
        if key in expected and outcome.get(key) != expected.get(key):
            return False
    if "invalid_reason" in expected and expected["invalid_reason"] not in outcome.get(
        "invalid_reasons", []
    ):
        return False
    return True


def plan(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "status": "PASS_SET_IDENTIFIED_PROSPECTUS_PLAN_NO_FIXTURE_RUN",
        "config_sha256": _sha256(config_path),
        "candidate_family": CANDIDATE_NAME,
        "analytic_fixture_count": len(config["analytic_fixtures"]),
        "opened_challenge_panel_replay": "FORBIDDEN_NOT_RUN",
        "candidate_selected": "NO",
        "scientific_effect": "NOT_ESTABLISHED",
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    outcomes = [
        _fixture_outcome(config, fixture) for fixture in config["analytic_fixtures"]
    ]
    matches = [
        _matches_expected(outcome, fixture["expected"])
        for outcome, fixture in zip(outcomes, config["analytic_fixtures"])
    ]
    require(all(matches), "analytic fixture definition conformance failed")
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS_ANALYTIC_DEFINITION_CONFORMANCE_NO_SELECTION",
        "completed_at_utc": _utc_now(),
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "candidate_family": CANDIDATE_NAME,
        "fixture_outcomes": outcomes,
        "fixture_counts": {
            "total": len(outcomes),
            "conformant": sum(matches),
            "valid": sum(bool(outcome["panel_valid"]) for outcome in outcomes),
            "invalid": sum(not bool(outcome["panel_valid"]) for outcome in outcomes),
            "point_identified": sum(
                outcome["identification_state"] == "POINT_IDENTIFIED_FIELD_VECTOR"
                for outcome in outcomes
            ),
            "set_identified": sum(
                str(outcome["identification_state"]).startswith("SET_IDENTIFIED")
                for outcome in outcomes
            ),
        },
        "disposition": {
            "candidate_admissibility": "NOT_EVALUATED",
            "candidate_selected": "NO",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "stochastic_candidate": "NOT_IMPLEMENTED",
            "fresh_blind_authorized": False,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        require(args.output is None, "plan forbids --output")
        result = plan(args.config)
    else:
        result = run(args.config)
        if args.output is not None:
            _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
