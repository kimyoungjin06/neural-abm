#!/usr/bin/env python3
"""Independently verify public stochastic set-development artifacts and metrics."""

from __future__ import annotations

import argparse
from collections import Counter
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
    "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml"
)
PRODUCER = ROOT / "scripts/develop_pivot_feedback_lean_h3_stochastic_set_v3.py"


class StochasticSetVerificationError(RuntimeError):
    """Raised when independent development verification detects drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StochasticSetVerificationError(message)


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


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth_valid = [row for row in rows if bool(row["truth_valid"])]
    truth_invalid = [row for row in rows if not bool(row["truth_valid"])]
    jointly_valid = [row for row in truth_valid if bool(row["valid"])]

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    contained = 0
    state_match = 0
    direction_match = 0
    for row in jointly_valid:
        estimated = row["field_intervals"]
        truth = row["truth_field_intervals"]
        if all(
            float(observed[0]) <= float(target[0])
            and float(observed[1]) >= float(target[1])
            for observed, target in zip(estimated, truth)
        ):
            contained += 1
        state_match += row["identification_state"] == row["truth_identification_state"]
        direction_match += row["direction_state"] == row["truth_direction_state"]
    reasons: Counter[str] = Counter()
    for row in rows:
        reason_counts = row.get("invalid_reason_counts")
        require(isinstance(reason_counts, dict), "invalid reason counts missing")
        reasons.update({str(key): int(value) for key, value in reason_counts.items()})
    return {
        "rows": len(rows),
        "truth_valid_estimator_valid_rate": rate(len(jointly_valid), len(truth_valid)),
        "truth_invalid_estimator_invalid_rate": rate(
            sum(not bool(row["valid"]) for row in truth_invalid),
            len(truth_invalid),
        ),
        "field_vector_interval_containment_among_jointly_valid": rate(
            contained,
            len(jointly_valid),
        ),
        "identification_state_match_among_jointly_valid": rate(
            state_match,
            len(jointly_valid),
        ),
        "direction_state_match_among_jointly_valid": rate(
            direction_match,
            len(jointly_valid),
        ),
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "denominators": {
            "truth_valid": len(truth_valid),
            "truth_invalid": len(truth_invalid),
            "jointly_valid": len(jointly_valid),
        },
    }


def verify(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(
        config.get("schema_version")
        == "pivot_feedback_lean_h3_stochastic_set_development.v3",
        "config schema drift",
    )
    panels_path = output_root / "public_panels.jsonl"
    truth_path = output_root / "public_truth.jsonl"
    generation_path = output_root / "generation_receipt.json"
    estimates_path = output_root / "candidate_estimates.jsonl"
    estimation_path = output_root / "estimation_receipt.json"
    report_path = output_root / "development_report.json"
    evaluation_path = output_root / "evaluation_receipt.json"
    generation = _read_json(generation_path)
    estimation = _read_json(estimation_path)
    report = _read_json(report_path)
    evaluation = _read_json(evaluation_path)
    require(generation.get("config_sha256") == _sha256(config_path), "generation config drift")
    require(generation.get("producer_sha256") == _sha256(PRODUCER), "generation producer drift")
    require(generation["panels"].get("sha256") == _sha256(panels_path), "panel drift")
    require(generation["truth"].get("sha256") == _sha256(truth_path), "truth drift")
    require(estimation.get("config_sha256") == _sha256(config_path), "estimation config drift")
    require(estimation.get("producer_sha256") == _sha256(PRODUCER), "estimation producer drift")
    require(estimation.get("truth_files_read") == 0, "estimation truth read drift")
    require(estimation["estimates"].get("sha256") == _sha256(estimates_path), "estimate drift")
    expected_bindings = {
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(PRODUCER),
        "estimation_receipt_sha256": _sha256(estimation_path),
        "truth_sha256": _sha256(truth_path),
        "estimates_sha256": _sha256(estimates_path),
        "report_sha256": _sha256(report_path),
    }
    require(evaluation.get("bindings") == expected_bindings, "evaluation binding drift")
    truth_rows = _read_jsonl(truth_path)
    estimate_rows = _read_jsonl(estimates_path)
    truth_map: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in truth_rows:
        key = (str(row["scenario_id"]), int(row["replicate"]))
        require(key not in truth_map, f"duplicate truth: {key}")
        truth_map[key] = row
    joined: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for estimate_row in estimate_rows:
        key = (str(estimate_row["scenario_id"]), int(estimate_row["replicate"]))
        require(key not in seen, f"duplicate estimate: {key}")
        seen.add(key)
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
    require(len(joined) == len(truth_rows), "cardinality drift")
    overall = _metric_block(joined)
    require(report.get("metrics") == overall, "overall metric drift")
    groups = sorted({str(row["group"]) for row in joined})
    group_metrics = {
        group: _metric_block([row for row in joined if row["group"] == group])
        for group in groups
    }
    require(report.get("group_metrics") == group_metrics, "group metric drift")
    scenarios = sorted({str(row["scenario_id"]) for row in joined})
    scenario_metrics = {
        scenario: _metric_block(
            [row for row in joined if row["scenario_id"] == scenario]
        )
        for scenario in scenarios
    }
    require(report.get("scenario_metrics") == scenario_metrics, "scenario metric drift")
    require(
        report.get("disposition")
        == {
            "candidate_admissibility": "NOT_EVALUATED",
            "candidate_selected": "NO",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "fresh_blind_authorized": False,
            "opened_challenge_reuse": "FORBIDDEN",
        },
        "disposition drift",
    )
    access = report.get("access_accounting")
    require(isinstance(access, dict) and set(access.values()) == {0}, "access drift")
    return {
        "schema_version": "pivot_feedback_lean_h3_stochastic_set_verification.v3",
        "status": "PASS_INDEPENDENT_PUBLIC_DEVELOPMENT_BINDING_AND_METRIC_REPLAY",
        "verified_at_utc": _utc_now(),
        "bindings": {
            **expected_bindings,
            "generation_receipt_sha256": _sha256(generation_path),
            "evaluation_receipt_sha256": _sha256(evaluation_path),
        },
        "replay_counts": {
            "panels": len(_read_jsonl(panels_path)),
            "truth_rows": len(truth_rows),
            "estimate_rows": len(estimate_rows),
            "groups": len(groups),
            "scenarios": len(scenarios),
        },
        "metrics": overall,
        "group_metrics": group_metrics,
        "candidate_admissibility": "NOT_EVALUATED",
        "candidate_selected": "NO",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "fresh_blind_authorized": False,
        "access_accounting": access,
        "scientific_effect": "NOT_ESTABLISHED",
        "claim_boundary": (
            "This replay verifies artifact bindings and descriptive metrics for "
            "a public nonblind synthetic development grid. It does not replay "
            "posterior draws, validate empirical dependence, establish "
            "admissibility, select an estimator, or authorize a fresh blind."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(args.config, args.output_root)
    if args.output is not None:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
