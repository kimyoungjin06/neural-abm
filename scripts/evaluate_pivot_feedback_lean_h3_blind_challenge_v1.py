#!/usr/bin/env python3
"""Unblind the frozen lean-H3 synthetic challenge and apply its rubric once."""

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
    "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml"
)
EXPECTED_ESTIMATORS = ("raw_earliest_argmax", "kernel_earliest_argmax")


class BlindEvaluationError(RuntimeError):
    """Raised when the one-time unblinding contract cannot be verified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BlindEvaluationError(message)


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


def _write_json(path: Path, payload: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"mode drift: {path}")


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
    require(
        config.get("schema_version") == "pivot_feedback_lean_h3_blind_challenge.v1",
        "config schema drift",
    )
    rubric = config.get("admissibility_rubric")
    require(isinstance(rubric, dict), "admissibility rubric missing")
    threshold_keys = (
        "minimum_valid_rate",
        "minimum_meaningful_positive_detection_rate",
        "maximum_nonpositive_directional_false_positive_rate",
        "minimum_identifiable_two_sided_coverage",
        "minimum_worst_family_identifiable_two_sided_coverage",
        "maximum_identifiable_sign_reversal_rate",
        "maximum_nonidentifiable_pass_rate",
        "meaningful_positive_truth_floor",
    )
    for key in threshold_keys:
        value = rubric.get(key)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"invalid rubric threshold: {key}",
        )
    return config


def _rate(numerator: int, denominator: int, label: str) -> float:
    require(denominator > 0, f"empty adjudication denominator: {label}")
    return numerator / denominator


def _joined_rows(
    estimates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    label_map: dict[tuple[str, int], Mapping[str, Any]] = {}
    for label in labels:
        key = (str(label["challenge_id"]), int(label["replicate"]))
        require(key not in label_map, f"duplicate private label: {key}")
        label_map[key] = label
    joined: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for estimate in estimates:
        key = (
            str(estimate["challenge_id"]),
            int(estimate["replicate"]),
            str(estimate["estimator"]),
        )
        require(key not in seen, f"duplicate estimate: {key}")
        seen.add(key)
        label_key = key[:2]
        require(label_key in label_map, f"missing private label: {label_key}")
        joined.append({**estimate, **label_map[label_key]})
    expected = len(label_map) * len(EXPECTED_ESTIMATORS)
    require(len(joined) == expected, "estimate/label cardinality mismatch")
    require({str(row["estimator"]) for row in joined} == set(EXPECTED_ESTIMATORS), "estimator set drift")
    return joined


def _estimator_metrics(
    rows: Sequence[Mapping[str, Any]],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    valid = [row for row in rows if bool(row["valid"])]
    identifiable = [row for row in valid if bool(row["truth_identifiable"])]
    nonidentifiable = [row for row in rows if not bool(row["truth_identifiable"])]
    positive_floor = float(rubric["meaningful_positive_truth_floor"])
    meaningful_positive = [
        row
        for row in rows
        if bool(row["truth_identifiable"])
        and float(row["truth_delta"]) >= positive_floor
    ]
    nonpositive = [
        row
        for row in rows
        if bool(row["truth_identifiable"])
        and float(row["truth_delta"]) <= 0.0
    ]
    signed = [
        row for row in identifiable if not np.isclose(float(row["truth_delta"]), 0.0)
    ]
    coverage_values = [
        float(row["two_sided_lower"])
        <= float(row["truth_delta"])
        <= float(row["two_sided_upper"])
        for row in identifiable
    ]
    sign_reversals = [
        (float(row["truth_delta"]) > 0.0 and float(row["point_estimate"]) < 0.0)
        or (float(row["truth_delta"]) < 0.0 and float(row["point_estimate"]) > 0.0)
        for row in signed
    ]
    family_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in identifiable:
        family_rows[str(row["generator_family"])].append(row)
    family_coverage = {
        family: _rate(
            sum(
                float(row["two_sided_lower"])
                <= float(row["truth_delta"])
                <= float(row["two_sided_upper"])
                for row in group
            ),
            len(group),
            f"{family}.coverage",
        )
        for family, group in sorted(family_rows.items())
    }
    metrics = {
        "rows": len(rows),
        "valid_rate": _rate(len(valid), len(rows), "valid_rate"),
        "meaningful_positive_detection_rate": _rate(
            sum(bool(row["passed"]) for row in meaningful_positive),
            len(meaningful_positive),
            "meaningful_positive_detection_rate",
        ),
        "nonpositive_directional_false_positive_rate": _rate(
            sum(bool(row["passed"]) for row in nonpositive),
            len(nonpositive),
            "nonpositive_directional_false_positive_rate",
        ),
        "identifiable_two_sided_coverage": _rate(
            sum(coverage_values), len(coverage_values), "identifiable_coverage"
        ),
        "worst_family_identifiable_two_sided_coverage": min(
            family_coverage.values()
        ),
        "identifiable_sign_reversal_rate": _rate(
            sum(sign_reversals), len(sign_reversals), "sign_reversal_rate"
        ),
        "nonidentifiable_pass_rate": _rate(
            sum(bool(row["passed"]) for row in nonidentifiable),
            len(nonidentifiable),
            "nonidentifiable_pass_rate",
        ),
        "mechanism_overreach_pass_rate": _rate(
            sum(
                bool(row["passed"]) and not bool(row["feedback_present"])
                for row in rows
            ),
            sum(not bool(row["feedback_present"]) for row in rows),
            "mechanism_overreach_pass_rate",
        ),
        "family_identifiable_two_sided_coverage": family_coverage,
        "denominators": {
            "valid": len(rows),
            "meaningful_positive": len(meaningful_positive),
            "nonpositive": len(nonpositive),
            "identifiable_coverage": len(coverage_values),
            "identifiable_signed": len(sign_reversals),
            "nonidentifiable": len(nonidentifiable),
        },
    }
    criteria = {
        "valid_rate": metrics["valid_rate"] >= float(rubric["minimum_valid_rate"]),
        "meaningful_positive_detection_rate": metrics[
            "meaningful_positive_detection_rate"
        ]
        >= float(rubric["minimum_meaningful_positive_detection_rate"]),
        "nonpositive_directional_false_positive_rate": metrics[
            "nonpositive_directional_false_positive_rate"
        ]
        <= float(rubric["maximum_nonpositive_directional_false_positive_rate"]),
        "identifiable_two_sided_coverage": metrics[
            "identifiable_two_sided_coverage"
        ]
        >= float(rubric["minimum_identifiable_two_sided_coverage"]),
        "worst_family_identifiable_two_sided_coverage": metrics[
            "worst_family_identifiable_two_sided_coverage"
        ]
        >= float(rubric["minimum_worst_family_identifiable_two_sided_coverage"]),
        "identifiable_sign_reversal_rate": metrics[
            "identifiable_sign_reversal_rate"
        ]
        <= float(rubric["maximum_identifiable_sign_reversal_rate"]),
        "nonidentifiable_pass_rate": metrics["nonidentifiable_pass_rate"]
        <= float(rubric["maximum_nonidentifiable_pass_rate"]),
    }
    return {**metrics, "criteria": criteria, "admissible": all(criteria.values())}


def adjudicate(
    estimates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    joined = _joined_rows(estimates, labels)
    metrics = {
        estimator: _estimator_metrics(
            [row for row in joined if row["estimator"] == estimator],
            rubric,
        )
        for estimator in EXPECTED_ESTIMATORS
    }
    eligible = [
        estimator for estimator in EXPECTED_ESTIMATORS if metrics[estimator]["admissible"]
    ]
    return {
        "estimator_metrics": metrics,
        "eligible_for_later_freeze": eligible,
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
    }


def evaluate(config_path: Path, challenge_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    panels_path = challenge_root / "challenge_panels.jsonl"
    labels_path = challenge_root / "private_labels.json"
    generation_path = challenge_root / "generation_receipt.json"
    estimates_path = challenge_root / "blind_estimates.jsonl"
    estimation_path = challenge_root / "blind_estimation_receipt.json"
    report_path = challenge_root / "unblinded_report.json"
    receipt_path = challenge_root / "unblinding_receipt.json"
    require(not report_path.exists(), "unblinded report already exists")
    require(not receipt_path.exists(), "unblinding receipt already exists")
    generation = _read_json(generation_path)
    estimation = _read_json(estimation_path)
    labels_payload = _read_json(labels_path)
    require(
        labels_payload.get("schema_version")
        == "pivot_feedback_lean_h3_blind_private_labels.v1",
        "private label schema drift",
    )
    require(
        generation.get("private_label_commitment_sha256") == _sha256(labels_path),
        "private label commitment mismatch",
    )
    require(
        estimation.get("status") == "PASS_BLIND_ESTIMATES_WRITTEN_BEFORE_UNBLINDING",
        "blind estimation incomplete",
    )
    require(estimation.get("config_sha256") == _sha256(config_path), "config binding drift")
    require(
        estimation.get("generation_receipt_sha256") == _sha256(generation_path),
        "generation receipt binding drift",
    )
    require(
        estimation.get("challenge_panels_sha256") == _sha256(panels_path),
        "panel binding drift",
    )
    require(
        estimation["blind_estimates"].get("sha256") == _sha256(estimates_path),
        "estimate binding drift",
    )
    estimates = _read_jsonl(estimates_path)
    labels = labels_payload.get("labels")
    require(isinstance(labels, list) and labels, "private labels missing")
    decision = adjudicate(estimates, labels, config["admissibility_rubric"])
    eligible = decision["eligible_for_later_freeze"]
    status = (
        "PASS_BLIND_CHALLENGE_AT_LEAST_ONE_ESTIMATOR_ELIGIBLE_NOT_SELECTED"
        if eligible
        else "BLOCKED_BLIND_CHALLENGE_NO_ESTIMATOR_ADMISSIBLE_NO_FREEZE"
    )
    report = {
        "schema_version": "pivot_feedback_lean_h3_blind_unblinded_report.v1",
        "status": status,
        "completed_at_utc": _utc_now(),
        "profile": generation["profile"],
        "precommit_git_commit": generation["precommit_git_commit"],
        "seed_hex": labels_payload["seed_hex"],
        "config_sha256": _sha256(config_path),
        "execution_counts": {
            "panel_rows": len(labels),
            "estimate_rows": len(estimates),
            "challenge_scenarios": len({str(row["challenge_id"]) for row in labels}),
            "generator_families": len(
                {str(row["generator_family"]) for row in labels}
            ),
        },
        "admissibility_rubric": config["admissibility_rubric"],
        **decision,
        "access_accounting": generation["access_accounting"],
        "authorization": {
            "one_time_unblinding": "COMPLETE",
            "post_unblind_retuning": "FORBIDDEN",
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
        "schema_version": "pivot_feedback_lean_h3_blind_unblinding_receipt.v1",
        "status": "PASS_ONE_TIME_UNBLINDING_RECEIPT_WRITTEN_LAST_SYNTHETIC_ONLY",
        "completed_at_utc": _utc_now(),
        "precommit_git_commit": generation["precommit_git_commit"],
        "bindings": {
            "config_sha256": _sha256(config_path),
            "generation_receipt_sha256": _sha256(generation_path),
            "challenge_panels_sha256": _sha256(panels_path),
            "private_labels_sha256": _sha256(labels_path),
            "blind_estimation_receipt_sha256": _sha256(estimation_path),
            "blind_estimates_sha256": _sha256(estimates_path),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "unblinded_report_sha256": _sha256(report_path),
        },
        "eligible_for_later_freeze": eligible,
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "access_accounting": generation["access_accounting"],
        "scientific_effect": "NOT_ESTABLISHED",
    }
    _write_json(receipt_path, receipt)
    return {
        "status": status,
        "eligible_for_later_freeze": eligible,
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "unblinded_report_sha256": _sha256(report_path),
        "unblinding_receipt_sha256": _sha256(receipt_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--challenge-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(args.config, args.challenge_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
