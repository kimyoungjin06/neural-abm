#!/usr/bin/env python3
"""Independently replay lean-H3 blind challenge bindings and adjudication."""

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
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml"
)
ESTIMATORS = ("raw_earliest_argmax", "kernel_earliest_argmax")
PANEL_KEYS = {"challenge_id", "replicate", "grid", "episodes"}
EPISODE_KEYS = {
    "field_id",
    "episode_id",
    "opportunities",
    "entries",
    "persistent",
}


class BlindVerificationError(RuntimeError):
    """Raised when independent blind challenge replay finds drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BlindVerificationError(message)


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


def _write_json(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"mode drift: {path}")


def _rate(numerator: int, denominator: int, label: str) -> float:
    require(denominator > 0, f"empty replay denominator: {label}")
    return numerator / denominator


def _replay_one(
    rows: Sequence[Mapping[str, Any]],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    valid = [row for row in rows if bool(row["valid"])]
    identifiable = [row for row in valid if bool(row["truth_identifiable"])]
    nonidentifiable = [row for row in rows if not bool(row["truth_identifiable"])]
    positive_floor = float(rubric["meaningful_positive_truth_floor"])
    meaningful = [
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
    coverage = [
        float(row["two_sided_lower"])
        <= float(row["truth_delta"])
        <= float(row["two_sided_upper"])
        for row in identifiable
    ]
    reversals = [
        (float(row["truth_delta"]) > 0.0 and float(row["point_estimate"]) < 0.0)
        or (float(row["truth_delta"]) < 0.0 and float(row["point_estimate"]) > 0.0)
        for row in signed
    ]
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in identifiable:
        by_family[str(row["generator_family"])].append(row)
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
        for family, group in sorted(by_family.items())
    }
    result: dict[str, Any] = {
        "rows": len(rows),
        "valid_rate": _rate(len(valid), len(rows), "valid_rate"),
        "meaningful_positive_detection_rate": _rate(
            sum(bool(row["passed"]) for row in meaningful),
            len(meaningful),
            "positive_detection",
        ),
        "nonpositive_directional_false_positive_rate": _rate(
            sum(bool(row["passed"]) for row in nonpositive),
            len(nonpositive),
            "nonpositive_false_positive",
        ),
        "identifiable_two_sided_coverage": _rate(
            sum(coverage), len(coverage), "coverage"
        ),
        "worst_family_identifiable_two_sided_coverage": min(
            family_coverage.values()
        ),
        "identifiable_sign_reversal_rate": _rate(
            sum(reversals), len(reversals), "sign_reversal"
        ),
        "nonidentifiable_pass_rate": _rate(
            sum(bool(row["passed"]) for row in nonidentifiable),
            len(nonidentifiable),
            "nonidentifiable_pass",
        ),
        "mechanism_overreach_pass_rate": _rate(
            sum(
                bool(row["passed"]) and not bool(row["feedback_present"])
                for row in rows
            ),
            sum(not bool(row["feedback_present"]) for row in rows),
            "mechanism_overreach",
        ),
        "family_identifiable_two_sided_coverage": family_coverage,
        "denominators": {
            "valid": len(rows),
            "meaningful_positive": len(meaningful),
            "nonpositive": len(nonpositive),
            "identifiable_coverage": len(coverage),
            "identifiable_signed": len(reversals),
            "nonidentifiable": len(nonidentifiable),
        },
    }
    criteria = {
        "valid_rate": result["valid_rate"] >= float(rubric["minimum_valid_rate"]),
        "meaningful_positive_detection_rate": result[
            "meaningful_positive_detection_rate"
        ]
        >= float(rubric["minimum_meaningful_positive_detection_rate"]),
        "nonpositive_directional_false_positive_rate": result[
            "nonpositive_directional_false_positive_rate"
        ]
        <= float(rubric["maximum_nonpositive_directional_false_positive_rate"]),
        "identifiable_two_sided_coverage": result["identifiable_two_sided_coverage"]
        >= float(rubric["minimum_identifiable_two_sided_coverage"]),
        "worst_family_identifiable_two_sided_coverage": result[
            "worst_family_identifiable_two_sided_coverage"
        ]
        >= float(rubric["minimum_worst_family_identifiable_two_sided_coverage"]),
        "identifiable_sign_reversal_rate": result["identifiable_sign_reversal_rate"]
        <= float(rubric["maximum_identifiable_sign_reversal_rate"]),
        "nonidentifiable_pass_rate": result["nonidentifiable_pass_rate"]
        <= float(rubric["maximum_nonidentifiable_pass_rate"]),
    }
    result["criteria"] = criteria
    result["admissible"] = all(criteria.values())
    return result


def verify(config_path: Path, challenge_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(
        config.get("schema_version") == "pivot_feedback_lean_h3_blind_challenge.v1",
        "config schema drift",
    )
    panels_path = challenge_root / "challenge_panels.jsonl"
    labels_path = challenge_root / "private_labels.json"
    generation_path = challenge_root / "generation_receipt.json"
    estimates_path = challenge_root / "blind_estimates.jsonl"
    estimation_path = challenge_root / "blind_estimation_receipt.json"
    report_path = challenge_root / "unblinded_report.json"
    receipt_path = challenge_root / "unblinding_receipt.json"
    generation = _read_json(generation_path)
    estimation = _read_json(estimation_path)
    labels_payload = _read_json(labels_path)
    report = _read_json(report_path)
    receipt = _read_json(receipt_path)
    bindings = receipt.get("bindings")
    require(isinstance(bindings, dict), "receipt bindings missing")
    expected_bindings = {
        "config_sha256": _sha256(config_path),
        "generation_receipt_sha256": _sha256(generation_path),
        "challenge_panels_sha256": _sha256(panels_path),
        "private_labels_sha256": _sha256(labels_path),
        "blind_estimation_receipt_sha256": _sha256(estimation_path),
        "blind_estimates_sha256": _sha256(estimates_path),
        "evaluator_sha256": _sha256(
            ROOT / "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py"
        ),
        "unblinded_report_sha256": _sha256(report_path),
    }
    require(bindings == expected_bindings, "unblinding receipt binding drift")
    require(
        generation.get("private_label_commitment_sha256") == _sha256(labels_path),
        "private label commitment drift",
    )
    require(
        estimation.get("private_label_files_opened") == 0,
        "blind estimator reports private label access",
    )
    require(
        estimation.get("completed_at_utc") <= report.get("completed_at_utc"),
        "unblinding predates blind estimation",
    )
    panels = _read_jsonl(panels_path)
    for panel in panels:
        require(set(panel) == PANEL_KEYS, "panel exposes non-blind fields")
        for episode in panel["episodes"]:
            require(set(episode) == EPISODE_KEYS, "episode exposes non-blind fields")
    estimates = _read_jsonl(estimates_path)
    labels = labels_payload.get("labels")
    require(isinstance(labels, list) and labels, "private labels missing")
    label_map: dict[tuple[str, int], Mapping[str, Any]] = {}
    for label in labels:
        key = (str(label["challenge_id"]), int(label["replicate"]))
        require(key not in label_map, f"duplicate label: {key}")
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
        require(key[:2] in label_map, f"missing label: {key[:2]}")
        joined.append({**estimate, **label_map[key[:2]]})
    require(len(joined) == len(labels) * len(ESTIMATORS), "join cardinality drift")
    rubric = config["admissibility_rubric"]
    replay_metrics = {
        estimator: _replay_one(
            [row for row in joined if row["estimator"] == estimator],
            rubric,
        )
        for estimator in ESTIMATORS
    }
    require(report.get("estimator_metrics") == replay_metrics, "metric replay drift")
    eligible = [
        estimator for estimator in ESTIMATORS if replay_metrics[estimator]["admissible"]
    ]
    require(
        report.get("eligible_for_later_freeze") == eligible,
        "eligibility replay drift",
    )
    expected_status = (
        "PASS_BLIND_CHALLENGE_AT_LEAST_ONE_ESTIMATOR_ELIGIBLE_NOT_SELECTED"
        if eligible
        else "BLOCKED_BLIND_CHALLENGE_NO_ESTIMATOR_ADMISSIBLE_NO_FREEZE"
    )
    require(report.get("status") == expected_status, "report status drift")
    require(receipt.get("eligible_for_later_freeze") == eligible, "receipt eligibility drift")
    access = report.get("access_accounting")
    require(isinstance(access, dict) and set(access.values()) == {0}, "access drift")
    require(report["authorization"].get("estimator_selected") == "NO", "selection drift")
    require(report["authorization"].get("support_rule_selected") == "NO", "support drift")
    require(report["authorization"].get("scientific_effect") == "NOT_ESTABLISHED", "claim drift")
    return {
        "schema_version": "pivot_feedback_lean_h3_blind_verification.v1",
        "status": "PASS_INDEPENDENT_BLIND_BINDING_AND_ADJUDICATION_REPLAY_SYNTHETIC_ONLY",
        "verified_at_utc": _utc_now(),
        "bindings": {
            **expected_bindings,
            "unblinding_receipt_sha256": _sha256(receipt_path),
        },
        "replay_counts": {
            "panel_rows": len(panels),
            "estimate_rows": len(estimates),
            "label_rows": len(labels),
        },
        "estimator_metrics": replay_metrics,
        "eligible_for_later_freeze": eligible,
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "access_accounting": access,
        "scientific_effect": "NOT_ESTABLISHED",
        "claim_boundary": (
            "This replay verifies artifact bindings, panel blindness fields, stored "
            "metrics, and the frozen rubric. It does not independently regenerate "
            "panels, create an independent-team blind, select an estimator or support "
            "rule, access empirical rows, or establish a scientific effect."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(args.config, args.challenge_root)
    if args.output is not None:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
