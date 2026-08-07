#!/usr/bin/env python3
"""Verify set-identified v2 analytic fixture result bindings and oracle states."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_set_identified_prospectus_v2.yaml"
)
PRODUCER = ROOT / "scripts/prototype_pivot_feedback_lean_h3_set_identified_v2.py"
RESULT_SCHEMA = "pivot_feedback_lean_h3_set_identified_fixture_result.v2"


class SetIdentifiedVerificationError(RuntimeError):
    """Raised when independent analytic fixture verification finds drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SetIdentifiedVerificationError(message)


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


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "result root must be mapping")
    return payload


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


def _matches_oracle(outcome: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in (
        "panel_valid",
        "identification_state",
        "field_intervals",
        "field_balanced_interval_hull",
        "direction_state",
    ):
        if key in expected:
            require(outcome.get(key) == expected.get(key), f"oracle drift: {key}")
    if "invalid_reason" in expected:
        require(
            expected["invalid_reason"] in outcome.get("invalid_reasons", []),
            "invalid reason oracle drift",
        )


def verify(config_path: Path, result_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(
        config.get("schema_version")
        == "pivot_feedback_lean_h3_set_identified_prospectus.v2",
        "config schema drift",
    )
    result = _read_json(result_path)
    require(result.get("schema_version") == RESULT_SCHEMA, "result schema drift")
    require(
        result.get("status") == "PASS_ANALYTIC_DEFINITION_CONFORMANCE_NO_SELECTION",
        "result status drift",
    )
    require(result.get("config_sha256") == _sha256(config_path), "config binding drift")
    require(result.get("producer_sha256") == _sha256(PRODUCER), "producer binding drift")
    outcomes = result.get("fixture_outcomes")
    fixtures = config.get("analytic_fixtures")
    require(isinstance(outcomes, list), "fixture outcomes missing")
    require(isinstance(fixtures, list), "fixture oracle missing")
    require(len(outcomes) == len(fixtures), "fixture cardinality drift")
    outcome_map: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        require(isinstance(outcome, dict), "fixture outcome must be mapping")
        fixture_id = str(outcome.get("fixture_id"))
        require(fixture_id not in outcome_map, f"duplicate fixture: {fixture_id}")
        outcome_map[fixture_id] = outcome
    for fixture in fixtures:
        require(isinstance(fixture, dict), "fixture oracle must be mapping")
        fixture_id = str(fixture.get("fixture_id"))
        require(fixture_id in outcome_map, f"missing fixture: {fixture_id}")
        expected = fixture.get("expected")
        require(isinstance(expected, dict), f"expected oracle missing: {fixture_id}")
        _matches_oracle(outcome_map[fixture_id], expected)
    disposition = result.get("disposition")
    require(
        disposition
        == {
            "candidate_admissibility": "NOT_EVALUATED",
            "candidate_selected": "NO",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "stochastic_candidate": "NOT_IMPLEMENTED",
            "fresh_blind_authorized": False,
        },
        "disposition drift",
    )
    access = result.get("access_accounting")
    require(isinstance(access, dict) and set(access.values()) == {0}, "access drift")
    return {
        "schema_version": "pivot_feedback_lean_h3_set_identified_verification.v2",
        "status": "PASS_INDEPENDENT_ANALYTIC_FIXTURE_ORACLE_REPLAY_NO_SELECTION",
        "verified_at_utc": _utc_now(),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(PRODUCER),
            "result_sha256": _sha256(result_path),
        },
        "fixture_counts": result["fixture_counts"],
        "candidate_admissibility": "NOT_EVALUATED",
        "candidate_selected": "NO",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "fresh_blind_authorized": False,
        "access_accounting": access,
        "scientific_effect": "NOT_ESTABLISHED",
        "claim_boundary": (
            "This verifier replays config-bound analytic fixture oracles and "
            "artifact bindings. It does not validate the practical margin, "
            "implement stochastic uncertainty, use opened challenge or empirical "
            "rows, select an estimator, or establish a scientific effect."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(args.config, args.result)
    if args.output is not None:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
