#!/usr/bin/env python3
"""Independently replay lean-H3 result-space aggregates and counterexamples."""

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
    "experiments/configs/pivot_feedback_lean_h3_result_space_v1.yaml"
)
DEFAULT_PRODUCER = ROOT / "scripts/simulate_pivot_feedback_lean_h3_result_space_v1.py"
EXPECTED_REPORT_SCHEMA = "pivot_feedback_lean_h3_result_space_report.v1"
EXPECTED_RECEIPT_SCHEMA = "pivot_feedback_lean_h3_result_space_receipt.v1"
EXPECTED_ESTIMATORS = {
    "raw_earliest_argmax",
    "kernel_earliest_argmax",
}
EXPECTED_ARTIFACTS = {
    "scenario_manifest.jsonl",
    "replicate_estimates.jsonl",
    "counterexamples.json",
    "result_space_report.json",
}
FLAG_WEIGHTS = {
    "directional_false_positive": 4,
    "delayed_feedback_false_negative": 1,
    "sign_reversal": 4,
    "confident_wrong_interval": 5,
    "nonidentifiable_truth_pass": 3,
    "mechanism_overreach_risk": 1,
    "aggregation_paradox": 2,
    "estimator_disagreement": 2,
}


class ResultSpaceVerificationError(RuntimeError):
    """Raised when an independent result-space replay fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultSpaceVerificationError(message)


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


def _load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            require(isinstance(payload, dict), f"non-mapping row at {path}:{line_number}")
            rows.append(payload)
    return rows


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
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
    require(stat.S_IMODE(path.stat().st_mode) == 0o600, "verification mode drift")


def _expected_flags(
    row: Mapping[str, Any],
    *,
    minimum_point_gap: float,
) -> dict[str, bool]:
    valid = bool(row["valid"])
    passed = bool(row["passed"])
    identifiable = bool(row["truth_identifiable"])
    feedback = bool(row["feedback_present"])
    truth = float(row["truth_delta"])
    point = row["point_estimate"]
    sign_reversal = bool(
        valid
        and identifiable
        and point is not None
        and (
            (truth > 0.0 and float(point) < 0.0)
            or (truth < 0.0 and float(point) > 0.0)
        )
    )
    confident_wrong = bool(
        valid
        and identifiable
        and not (
            float(row["two_sided_lower"])
            <= truth
            <= float(row["two_sided_upper"])
        )
    )
    aggregation_paradox = bool(
        valid
        and point is not None
        and float(point) > 0.0
        and row["episode_nonpositive_share"] is not None
        and float(row["episode_nonpositive_share"]) > 0.5
    )
    return {
        "directional_false_positive": bool(
            identifiable and truth <= 0.0 and passed
        ),
        "delayed_feedback_false_negative": bool(
            feedback
            and identifiable
            and truth >= minimum_point_gap
            and not passed
        ),
        "sign_reversal": sign_reversal,
        "confident_wrong_interval": confident_wrong,
        "nonidentifiable_truth_pass": bool(not identifiable and passed),
        "mechanism_overreach_risk": bool(not feedback and passed),
        "aggregation_paradox": aggregation_paradox,
        "estimator_disagreement": False,
    }


def _verify_rows(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_point_gap: float,
) -> None:
    paired: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        require(row["estimator"] in EXPECTED_ESTIMATORS, "estimator surface drift")
        require(set(row["flags"]) == set(FLAG_WEIGHTS), "flag surface drift")
        paired[(str(row["scenario_id"]), int(row["replicate"]))].append(row)
    for key, pair in paired.items():
        require(len(pair) == 2, f"unpaired estimator rows: {key}")
        require({row["estimator"] for row in pair} == EXPECTED_ESTIMATORS, f"estimator pair drift: {key}")
        first, second = pair
        first_sign = (
            0
            if first["point_estimate"] is None
            else int(np.sign(float(first["point_estimate"])))
        )
        second_sign = (
            0
            if second["point_estimate"] is None
            else int(np.sign(float(second["point_estimate"])))
        )
        disagreement = bool(
            bool(first["passed"]) != bool(second["passed"])
            or first_sign != second_sign
        )
        for row in pair:
            expected = _expected_flags(
                row,
                minimum_point_gap=minimum_point_gap,
            )
            expected["estimator_disagreement"] = disagreement
            require(row["flags"] == expected, f"flag replay mismatch: {key}")
            expected_score = sum(
                FLAG_WEIGHTS[name]
                for name, active in expected.items()
                if active
            )
            require(row["surprise_score"] == expected_score, f"score mismatch: {key}")


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group in sorted(grouped.items(), key=lambda item: item[0]):
        valid = [row for row in group if row["valid"]]
        identifiable = [row for row in valid if row["truth_identifiable"]]
        biases = [
            float(row["point_estimate"]) - float(row["truth_delta"])
            for row in identifiable
        ]
        coverage = [
            float(row["two_sided_lower"])
            <= float(row["truth_delta"])
            <= float(row["two_sided_upper"])
            for row in identifiable
        ]
        summary = {key: value for key, value in zip(keys, group_key, strict=True)}
        summary.update(
            {
                "replicate_rows": len(group),
                "valid_rate": len(valid) / len(group),
                "pass_rate": sum(bool(row["passed"]) for row in group) / len(group),
                "mean_point_estimate": _safe_mean(
                    [float(row["point_estimate"]) for row in valid]
                ),
                "mean_truth_delta": _safe_mean(
                    [float(row["truth_delta"]) for row in identifiable]
                ),
                "mean_bias": _safe_mean(biases),
                "two_sided_coverage": _safe_mean([float(value) for value in coverage]),
                "flag_rates": {
                    flag: sum(bool(row["flags"][flag]) for row in group) / len(group)
                    for flag in FLAG_WEIGHTS
                },
            }
        )
        summaries.append(summary)
    return summaries


def _expected_counterexamples(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["surprise_score"]) > 0]
    selected.sort(
        key=lambda row: (
            -int(row["surprise_score"]),
            str(row["scenario_id"]),
            int(row["replicate"]),
            str(row["estimator"]),
        )
    )
    return [dict(row) for row in selected[:200]]


def verify(
    *,
    config_path: Path,
    producer_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    require(stat.S_IMODE(output_root.stat().st_mode) == 0o700, "output root mode drift")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "config root drift")
    receipt = _load_json(output_root / "receipt.json")
    require(receipt["schema_version"] == EXPECTED_RECEIPT_SCHEMA, "receipt schema drift")
    profile_name = receipt["profile"]
    require(
        profile_name in config["design"]["profiles"],
        "receipt profile is not declared",
    )
    require(receipt["config"]["sha256"] == _sha256(config_path), "config hash drift")
    require(receipt["producer"]["sha256"] == _sha256(producer_path), "producer hash drift")
    inventory = {item["basename"]: item for item in receipt["artifacts"]}
    require(set(inventory) == EXPECTED_ARTIFACTS, "receipt artifact surface drift")
    for basename, binding in inventory.items():
        path = output_root / basename
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"mode drift: {basename}")
        require(path.stat().st_size == binding["bytes"], f"size drift: {basename}")
        require(_sha256(path) == binding["sha256"], f"hash drift: {basename}")

    manifest = _load_jsonl(output_root / "scenario_manifest.jsonl")
    rows = _load_jsonl(output_root / "replicate_estimates.jsonl")
    counterexamples = _load_json(output_root / "counterexamples.json")
    report = _load_json(output_root / "result_space_report.json")
    require(report["schema_version"] == EXPECTED_REPORT_SCHEMA, "report schema drift")
    profile = config["design"]["profiles"][profile_name]
    canonical_count = len(config["canonical_scenarios"])
    adversarial_count = int(profile["adversarial_scenarios"])
    require(len(manifest) == canonical_count + adversarial_count, "manifest count drift")
    expected_rows = 2 * (
        canonical_count * int(profile["canonical_replicates"])
        + adversarial_count * int(profile["adversarial_replicates"])
    )
    require(len(rows) == expected_rows, "replicate row-count drift")
    require(
        {row["mechanism_family"] for row in manifest}
        == set(config["mechanism_families"]),
        "mechanism coverage drift",
    )
    _verify_rows(
        rows,
        minimum_point_gap=float(
            config["design"]["decision_rule"][
                "minimum_point_gap_cohort_years"
            ]
        ),
    )

    expected_flags = {
        flag: sum(bool(row["flags"][flag]) for row in rows)
        for flag in FLAG_WEIGHTS
    }
    require(report["flag_counts"] == expected_flags, "flag aggregate mismatch")
    canonical_rows = [row for row in rows if row["scenario_kind"] == "canonical"]
    expected_canonical = _summaries(
        canonical_rows,
        keys=("scenario_id", "mechanism_family", "estimator"),
    )
    require(
        report["canonical_operating_characteristics"] == expected_canonical,
        "canonical summary mismatch",
    )
    expected_mechanisms = _summaries(
        rows,
        keys=("scenario_kind", "mechanism_family", "estimator"),
    )
    require(
        report["mechanism_operating_characteristics"] == expected_mechanisms,
        "mechanism summary mismatch",
    )
    expected_counterexamples = _expected_counterexamples(rows)
    require(counterexamples == expected_counterexamples, "counterexample ranking mismatch")
    require(
        report["access_accounting"]
        == {
            "source_or_empirical_rows_read": 0,
            "current_g1_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "models_fitted_to_empirical_data": 0,
        },
        "access accounting drift",
    )

    canonical_by_key = {
        (item["scenario_id"], item["estimator"]): item
        for item in expected_canonical
    }
    return {
        "schema_version": "pivot_feedback_lean_h3_result_space_verification.v1",
        "status": "PASS_INDEPENDENT_AGGREGATE_FLAG_AND_COUNTEREXAMPLE_REPLAY_SYNTHETIC_ONLY",
        "verified_at_utc": _utc_now(),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(producer_path),
            "receipt_sha256": _sha256(output_root / "receipt.json"),
            "report_sha256": _sha256(output_root / "result_space_report.json"),
            "scenario_manifest_sha256": _sha256(
                output_root / "scenario_manifest.jsonl"
            ),
            "replicate_estimates_sha256": _sha256(
                output_root / "replicate_estimates.jsonl"
            ),
            "counterexamples_sha256": _sha256(output_root / "counterexamples.json"),
        },
        "replay_counts": {
            "scenario_rows": len(manifest),
            "replicate_estimate_rows": len(rows),
            "counterexample_rows_total": sum(
                int(row["surprise_score"]) > 0 for row in rows
            ),
            "counterexample_rows_retained": len(counterexamples),
            "flag_counts": expected_flags,
        },
        "canonical_findings": {
            "sharp_positive_raw": canonical_by_key[
                ("delayed_gap_plus_one_sharp", "raw_earliest_argmax")
            ],
            "synchronous_null_raw": canonical_by_key[
                ("synchronous_null", "raw_earliest_argmax")
            ],
            "broad_positive_kernel": canonical_by_key[
                ("delayed_gap_plus_two_broad", "kernel_earliest_argmax")
            ],
            "bimodal_recruitment_raw": canonical_by_key[
                ("recruitment_bimodal", "raw_earliest_argmax")
            ],
            "bimodal_recruitment_kernel": canonical_by_key[
                ("recruitment_bimodal", "kernel_earliest_argmax")
            ],
            "measurement_mimic_raw": canonical_by_key[
                ("measurement_shift_mimics_positive_gap", "raw_earliest_argmax")
            ],
        },
        "access_accounting": report["access_accounting"],
        "authorization": {
            "synthetic_result_space_replay": "COMPLETE",
            "estimator_selected": "NO",
            "support_rule_selected": "NO",
            "synthetic_blind_challenge": "DEFERRED",
            "empirical_row_access": "FORBIDDEN_NOT_RUN",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "This independent replay validates stored synthetic aggregates, flags, "
            "and counterexample ranking. It does not independently regenerate panels, "
            "select an estimator or support rule, access empirical rows, validate a "
            "mechanism, or establish a scientific effect."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--producer", type=Path, default=DEFAULT_PRODUCER)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        producer_path=args.producer,
        output_root=args.output_root,
    )
    if args.verification_output is not None:
        _write_create_only(args.verification_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
