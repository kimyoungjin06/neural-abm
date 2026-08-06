#!/usr/bin/env python3
"""Independently replay peak-set development diagnostic bindings and metrics."""

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

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_peak_set_development_v1.yaml"
)


class PeakSetVerificationError(RuntimeError):
    """Raised when independent development diagnostic replay finds drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PeakSetVerificationError(message)


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
            require(isinstance(payload, dict), f"invalid row {line_number}: {path}")
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


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        reason_counts = row.get("invalid_reason_counts")
        require(isinstance(reason_counts, dict), "invalid reason counts missing")
        reasons.update({str(key): int(value) for key, value in reason_counts.items()})

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


def verify(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root must be mapping")
    require(
        config.get("schema_version")
        == "pivot_feedback_lean_h3_peak_set_development.v1",
        "config schema drift",
    )
    source_root = Path(config["source_diagnostic"]["root"])
    source_bindings = {
        "challenge_panels_sha256": source_root / "challenge_panels.jsonl",
        "private_labels_sha256": source_root / "private_labels.json",
        "frozen_blind_estimates_sha256": source_root / "blind_estimates.jsonl",
        "frozen_unblinded_report_sha256": source_root / "unblinded_report.json",
    }
    for key, path in source_bindings.items():
        require(
            _sha256(path) == config["source_diagnostic"].get(key),
            f"source binding drift: {key}",
        )
    labels_path = source_root / "private_labels.json"
    frozen_report_path = source_root / "unblinded_report.json"
    estimates_path = output_root / "candidate_estimates.jsonl"
    estimation_receipt_path = output_root / "estimation_receipt.json"
    report_path = output_root / "diagnostic_report.json"
    receipt_path = output_root / "diagnostic_receipt.json"
    estimation = _read_json(estimation_receipt_path)
    report = _read_json(report_path)
    receipt = _read_json(receipt_path)
    bindings = receipt.get("bindings")
    require(isinstance(bindings, dict), "diagnostic receipt bindings missing")
    expected_bindings = {
        "config_sha256": _sha256(config_path),
        "developer_sha256": _sha256(
            ROOT / "scripts/develop_pivot_feedback_lean_h3_peak_set_estimator_v1.py"
        ),
        "estimation_receipt_sha256": _sha256(estimation_receipt_path),
        "candidate_estimates_sha256": _sha256(estimates_path),
        "opened_private_labels_sha256": _sha256(labels_path),
        "diagnostic_report_sha256": _sha256(report_path),
    }
    require(bindings == expected_bindings, "diagnostic receipt binding drift")
    require(estimation.get("opened_private_label_files_read") == 0, "estimation label access drift")
    require(
        estimation.get("completed_at_utc") <= report.get("completed_at_utc"),
        "diagnostic predates candidate estimation",
    )
    labels_payload = _read_json(labels_path)
    labels = labels_payload.get("labels")
    require(isinstance(labels, list) and labels, "opened labels missing")
    label_map: dict[tuple[str, int], Mapping[str, Any]] = {}
    for label in labels:
        key = (str(label["challenge_id"]), int(label["replicate"]))
        require(key not in label_map, f"duplicate label: {key}")
        label_map[key] = label
    estimates = _read_jsonl(estimates_path)
    require(len(estimates) == len(label_map), "estimate/label cardinality drift")
    joined: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for estimate in estimates:
        key = (str(estimate["challenge_id"]), int(estimate["replicate"]))
        require(key not in seen, f"duplicate estimate: {key}")
        seen.add(key)
        require(key in label_map, f"missing label: {key}")
        joined.append({**estimate, **label_map[key]})
    replay = _metrics(joined)
    require(report.get("metrics") == replay, "overall diagnostic metric drift")
    families = sorted({str(row["generator_family"]) for row in joined})
    family_replay = {
        family: _metrics(
            [row for row in joined if str(row["generator_family"]) == family]
        )
        for family in families
    }
    require(report.get("family_metrics") == family_replay, "family metric drift")
    frozen_report = _read_json(frozen_report_path)
    require(
        report["frozen_v1_reference"].get("status") == frozen_report.get("status"),
        "frozen v1 status drift",
    )
    require(
        report["frozen_v1_reference"].get("estimator_metrics")
        == frozen_report.get("estimator_metrics"),
        "frozen v1 metric drift",
    )
    require(
        report["disposition"]
        == {
            "candidate_admissibility": "NOT_EVALUATED",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "opened_challenge_reuse_as_blind_confirmation": "FORBIDDEN",
            "fresh_blind_promotion": (
                "REQUIRED_SEPARATE_PRECOMMIT_AND_DISJOINT_GENERATOR"
            ),
        },
        "disposition drift",
    )
    access = report.get("access_accounting")
    require(isinstance(access, dict) and set(access.values()) == {0}, "access drift")
    return {
        "schema_version": "pivot_feedback_lean_h3_peak_set_development_verification.v1",
        "status": "PASS_INDEPENDENT_OPENED_DIAGNOSTIC_BINDING_AND_METRIC_REPLAY_NO_SELECTION",
        "verified_at_utc": _utc_now(),
        "bindings": {
            **expected_bindings,
            "diagnostic_receipt_sha256": _sha256(receipt_path),
        },
        "replay_counts": {
            "candidate_estimate_rows": len(estimates),
            "opened_label_rows": len(labels),
            "generator_families": len(families),
        },
        "metrics": replay,
        "family_metrics": family_replay,
        "candidate_admissibility": "NOT_EVALUATED",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "access_accounting": access,
        "scientific_effect": "NOT_ESTABLISHED",
        "claim_boundary": (
            "This replay verifies development artifact bindings and descriptive "
            "metrics on an already-opened synthetic challenge. It does not replay "
            "posterior draws, create blind evidence, select an estimator or support "
            "rule, authorize empirical access, or establish a scientific effect."
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
