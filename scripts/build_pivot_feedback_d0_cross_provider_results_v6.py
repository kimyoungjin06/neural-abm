#!/usr/bin/env python3
"""Validate both API judges and build the blinded v6 human review queue."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, ROUND_UP
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v6.yaml"
)
DEFAULT_V4_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
DEFAULT_V4_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
DEFAULT_V5_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
DEFAULT_V5_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729.verification-v5.json"
)
DEFAULT_V5_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "execution-20260729"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
)
STATUS = "PASS_VALID_CROSS_PROVIDER_RESULTS_HUMAN_REVIEW_PENDING"
LABELS = {"ON_TOPIC", "AMBIGUOUS", "OFF_TOPIC", "UNREVIEWABLE"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
PACKET_FIELDS = [
    "review_item_id",
    "target_topic_name",
    "title",
    "language",
    "work_type",
    "reviewer_label",
    "reviewer_notes",
]


class CrossProviderResultBuildError(RuntimeError):
    """Fail-closed result reduction error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderResultBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossProviderResultBuildError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CrossProviderResultBuildError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _safe_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            rows.append(_mapping(json.loads(line), f"{path.name}:{line_number}"))
        except json.JSONDecodeError as error:
            raise CrossProviderResultBuildError(
                f"invalid JSONL {path.name}:{line_number}: {error}"
            ) from error
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_decision(value: Any, label: str) -> dict[str, str]:
    decision = _mapping(value, label)
    require(
        set(decision) == {"confidence", "label", "rationale"},
        f"{label} keys differ",
    )
    require(decision["label"] in LABELS, f"{label} decision label differs")
    require(
        decision["confidence"] in CONFIDENCES,
        f"{label} confidence differs",
    )
    require(
        isinstance(decision["rationale"], str)
        and bool(decision["rationale"].strip()),
        f"{label} rationale missing",
    )
    return {
        "label": str(decision["label"]),
        "confidence": str(decision["confidence"]),
        "rationale": str(decision["rationale"]).strip(),
    }


def _parse_openai(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows = _read_jsonl(path)
    decisions: dict[str, dict[str, str]] = {}
    input_tokens = 0
    output_tokens = 0
    for row in rows:
        custom_id = row.get("custom_id")
        require(
            isinstance(custom_id, str) and len(custom_id) == 64,
            "OpenAI custom ID differs",
        )
        require(custom_id not in decisions, "duplicate OpenAI custom ID")
        require(row.get("error") is None, "OpenAI provider error present")
        response = _mapping(row.get("response"), "OpenAI response")
        require(response.get("status_code") == 200, "OpenAI HTTP status differs")
        body = _mapping(response.get("body"), "OpenAI response body")
        require(body.get("status") == "completed", "OpenAI response incomplete")
        require(not body.get("incomplete_details"), "OpenAI incomplete details present")
        usage = _mapping(body.get("usage"), "OpenAI usage")
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        texts = [
            content.get("text")
            for item in body.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        require(len(texts) == 1 and isinstance(texts[0], str), "OpenAI output text differs")
        try:
            raw_decision = json.loads(texts[0])
        except json.JSONDecodeError as error:
            raise CrossProviderResultBuildError(
                f"invalid OpenAI structured output: {error}"
            ) from error
        decisions[custom_id] = _validate_decision(raw_decision, "OpenAI decision")
    require(len(rows) == len(decisions) == 384, "OpenAI row count differs")
    return decisions, {
        "rows": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _parse_claude(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows = _read_jsonl(path)
    decisions: dict[str, dict[str, str]] = {}
    input_tokens = 0
    output_tokens = 0
    for row in rows:
        custom_id = row.get("custom_id")
        require(
            isinstance(custom_id, str) and len(custom_id) == 64,
            "Claude custom ID differs",
        )
        require(custom_id not in decisions, "duplicate Claude custom ID")
        result = _mapping(row.get("result"), "Claude result")
        require(result.get("type") == "succeeded", "Claude result failed")
        message = _mapping(result.get("message"), "Claude message")
        usage = _mapping(message.get("usage"), "Claude usage")
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        content = message.get("content")
        require(isinstance(content, list) and len(content) == 1, "Claude content differs")
        text_item = _mapping(content[0], "Claude text item")
        require(
            text_item.get("type") == "text"
            and isinstance(text_item.get("text"), str),
            "Claude output text differs",
        )
        try:
            raw_decision = json.loads(str(text_item["text"]))
        except json.JSONDecodeError as error:
            raise CrossProviderResultBuildError(
                f"invalid Claude structured output: {error}"
            ) from error
        decisions[custom_id] = _validate_decision(raw_decision, "Claude decision")
    require(len(rows) == len(decisions) == 384, "Claude row count differs")
    return decisions, {
        "rows": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _artifact_tree(artifacts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(artifacts, key=lambda value: str(value["relative_path"])):
        digest.update(str(item["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _cleanup_temporary_directory(
    temporary: Path | None,
    *,
    output: Path,
) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary path must be absolute")
    require(output.is_absolute(), "output path must be absolute")
    require(temporary != output, "temporary path equals output")
    require(temporary.parent == output.parent, "temporary path not output sibling")
    require(
        temporary.name.startswith(f".{output.name}.tmp-"),
        "temporary prefix differs",
    )
    if temporary.exists():
        require(
            temporary.is_dir() and not temporary.is_symlink(),
            "temporary path invalid",
        )
        shutil.rmtree(temporary)


def _review_id(api_item_id: str, reviewer: int) -> str:
    return hashlib.sha256(
        f"pivot-feedback-d0-v6|reviewer-{reviewer}|{api_item_id}".encode()
    ).hexdigest()


def _money(
    *,
    input_tokens: int,
    output_tokens: int,
    rates: Mapping[str, Any],
) -> Decimal:
    return (
        Decimal(input_tokens) * Decimal(str(rates["input"]))
        + Decimal(output_tokens) * Decimal(str(rates["output"]))
    ) / Decimal(1_000_000)


def build(
    *,
    config_path: Path,
    v4_prepared_root: Path,
    v4_execution_root: Path,
    v5_prepared_root: Path,
    v5_report: Path,
    v5_execution_root: Path,
    terminal_snapshot: Path,
    output: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = _load_yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v6",
        "config schema differs",
    )
    require(
        config.get("status")
        == "FROZEN_VALID_RESULT_REDUCTION_CONTRACT_HUMAN_REVIEW_PENDING",
        "config status differs",
    )
    parent = _mapping(config.get("contract_parent"), "contract parent")
    parent_path = _safe_repo_file(repo_root, parent.get("relative_path"), "parent config")
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA differs")
    require(
        _load_yaml(parent_path).get("schema_version")
        == parent.get("required_schema_version"),
        "parent schema differs",
    )
    inputs = _mapping(config.get("inputs"), "inputs")

    v4_prepared_root = v4_prepared_root.resolve(strict=True)
    v4_execution_root = v4_execution_root.resolve(strict=True)
    v5_prepared_root = v5_prepared_root.resolve(strict=True)
    v5_report = v5_report.resolve(strict=True)
    v5_execution_root = v5_execution_root.resolve(strict=True)
    terminal_snapshot = terminal_snapshot.resolve(strict=True)
    require(
        v4_prepared_root.name == inputs.get("v4_prepared_root_basename"),
        "v4 prepared root differs",
    )
    require(
        v4_execution_root.name == inputs.get("v4_execution_root_basename"),
        "v4 execution root differs",
    )
    require(
        v5_prepared_root.name == inputs.get("v5_prepared_root_basename"),
        "v5 prepared root differs",
    )
    require(
        v5_execution_root.name == inputs.get("v5_execution_root_basename"),
        "v5 execution root differs",
    )
    api_key_path = v4_prepared_root / "api_request_key.csv"
    calibration_key_path = v4_prepared_root / "human_calibration_key.csv"
    claude_path = v4_execution_root / "claude_results.jsonl"
    v4_submission_path = v4_execution_root / "cross_provider_submission.outputs.json"
    v5_preparation_path = v5_prepared_root / "openai_retry_preparation.outputs.json"
    v5_submission_path = v5_execution_root / "openai_retry_submission.outputs.json"
    openai_path = v5_execution_root / "openai_retry_output.jsonl"
    bindings = {
        api_key_path: inputs["api_request_key_sha256"],
        calibration_key_path: inputs["human_calibration_key_sha256"],
        v4_submission_path: inputs["v4_submission_receipt_sha256"],
        claude_path: inputs["claude_results_sha256"],
        v5_preparation_path: inputs["v5_preparation_receipt_sha256"],
        v5_report: inputs["v5_verification_report_sha256"],
        v5_submission_path: inputs["v5_submission_receipt_sha256"],
        openai_path: inputs["openai_retry_output_sha256"],
        terminal_snapshot: inputs["terminal_snapshot_sha256"],
    }
    for path, expected_sha in bindings.items():
        require(_sha256(path) == expected_sha, f"input SHA differs: {path.name}")
    v4_submission = _load_json(v4_submission_path)
    v5_submission = _load_json(v5_submission_path)
    snapshot = _load_json(terminal_snapshot)
    require(
        _mapping(v4_submission["claude"], "Claude submission").get("batch_id")
        == inputs.get("claude_batch_id"),
        "Claude batch ID differs",
    )
    require(
        _mapping(v5_submission["openai"], "OpenAI submission").get("batch_id")
        == inputs.get("openai_batch_id"),
        "OpenAI batch ID differs",
    )
    require(
        _mapping(snapshot["openai"], "terminal OpenAI").get("status") == "completed",
        "OpenAI snapshot not completed",
    )
    require(snapshot.get("terminal") is True, "terminal snapshot differs")

    openai, openai_usage = _parse_openai(openai_path)
    claude, claude_usage = _parse_claude(claude_path)
    require(set(openai) == set(claude), "provider custom-ID universes differ")
    key_rows = _read_csv(api_key_path)
    require(len(key_rows) == 384, "API key row count differs")
    key_by_id = {row["api_item_id"]: row for row in key_rows}
    require(len(key_by_id) == 384, "API key IDs not unique")
    require(set(key_by_id) == set(openai), "API key universe differs")
    calibration_rows = _read_csv(calibration_key_path)
    calibration_ids = {row["api_item_id"] for row in calibration_rows}
    require(
        len(calibration_rows) == len(calibration_ids) == 96,
        "calibration rows differ",
    )
    require(calibration_ids <= set(openai), "calibration universe differs")

    label_pairs: Counter[tuple[str, str]] = Counter()
    confidence_pairs: Counter[tuple[str, str]] = Counter()
    openai_labels: Counter[str] = Counter()
    claude_labels: Counter[str] = Counter()
    openai_confidences: Counter[str] = Counter()
    claude_confidences: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    auto_accept: list[dict[str, Any]] = []
    queue_ids = set(calibration_ids)
    escalation_ids: set[str] = set()
    for api_item_id in sorted(openai, key=lambda value: int(key_by_id[value]["source_row_number"])):
        source = key_by_id[api_item_id]
        oa = openai[api_item_id]
        cl = claude[api_item_id]
        openai_labels[oa["label"]] += 1
        claude_labels[cl["label"]] += 1
        openai_confidences[oa["confidence"]] += 1
        claude_confidences[cl["confidence"]] += 1
        label_pairs[(oa["label"], cl["label"])] += 1
        confidence_pairs[(oa["confidence"], cl["confidence"])] += 1
        reasons = []
        if oa["label"] != cl["label"]:
            reasons.append("LABEL_DISAGREEMENT")
        if oa["label"] != "ON_TOPIC":
            reasons.append("OPENAI_NOT_ON_TOPIC")
        if cl["label"] != "ON_TOPIC":
            reasons.append("CLAUDE_NOT_ON_TOPIC")
        if oa["confidence"] != "HIGH":
            reasons.append("OPENAI_NOT_HIGH_CONFIDENCE")
        if cl["confidence"] != "HIGH":
            reasons.append("CLAUDE_NOT_HIGH_CONFIDENCE")
        for reason in reasons:
            reason_counts[reason] += 1
        if reasons:
            escalation_ids.add(api_item_id)
            queue_ids.add(api_item_id)
        dual_high_on_topic = (
            oa["label"] == cl["label"] == "ON_TOPIC"
            and oa["confidence"] == cl["confidence"] == "HIGH"
        )
        decision = {
            **source,
            "openai_label": oa["label"],
            "openai_confidence": oa["confidence"],
            "openai_rationale": oa["rationale"],
            "claude_label": cl["label"],
            "claude_confidence": cl["confidence"],
            "claude_rationale": cl["rationale"],
            "label_agreement": str(oa["label"] == cl["label"]).lower(),
            "dual_high_confidence_on_topic": str(dual_high_on_topic).lower(),
            "base_calibration": str(api_item_id in calibration_ids).lower(),
            "api_escalated": str(bool(reasons)).lower(),
            "escalation_reasons": "|".join(reasons),
        }
        decisions.append(decision)
        if dual_high_on_topic:
            auto_accept.append(
                {
                    **source,
                    "openai_label": oa["label"],
                    "openai_confidence": oa["confidence"],
                    "claude_label": cl["label"],
                    "claude_confidence": cl["confidence"],
                }
            )

    agreement = sum(count for (left, right), count in label_pairs.items() if left == right)
    expected = _mapping(config.get("expected_reduction"), "expected reduction")
    reduction = {
        "label_agreement_rows": agreement,
        "label_disagreement_rows": 384 - agreement,
        "api_escalation_rows": len(escalation_ids),
        "base_calibration_rows": len(calibration_ids),
        "calibration_escalation_overlap_rows": len(calibration_ids & escalation_ids),
        "blinded_human_queue_rows": len(queue_ids),
        "dual_high_confidence_on_topic_rows": len(auto_accept),
    }
    require(reduction == expected, "reduction counts differ")

    cost = _mapping(config.get("cost_control"), "cost control")
    openai_cost = _money(
        input_tokens=openai_usage["input_tokens"],
        output_tokens=openai_usage["output_tokens"],
        rates=_mapping(cost["openai_batch_price_per_million"], "OpenAI rates"),
    )
    claude_cost = _money(
        input_tokens=claude_usage["input_tokens"],
        output_tokens=claude_usage["output_tokens"],
        rates=_mapping(
            cost["claude_haiku_4_5_batch_price_per_million"],
            "Claude rates",
        ),
    )
    v4_actual = Decimal(str(cost["v4_actual_total_cost_usd"]))
    cumulative = v4_actual + openai_cost
    quant = Decimal("0.000001")
    cost_summary = {
        "openai_retry_input_tokens": openai_usage["input_tokens"],
        "openai_retry_output_tokens": openai_usage["output_tokens"],
        "openai_retry_actual_batch_cost_usd": str(
            openai_cost.quantize(quant, rounding=ROUND_UP)
        ),
        "claude_input_tokens": claude_usage["input_tokens"],
        "claude_output_tokens": claude_usage["output_tokens"],
        "claude_actual_batch_cost_usd": str(
            claude_cost.quantize(quant, rounding=ROUND_UP)
        ),
        "v4_actual_total_cost_usd": str(v4_actual.quantize(quant)),
        "cumulative_actual_cost_usd": str(
            cumulative.quantize(quant, rounding=ROUND_UP)
        ),
        "cumulative_authorized_maximum_usd": str(
            Decimal(str(cost["cumulative_authorized_maximum_usd"])).quantize(
                Decimal("0.00")
            )
        ),
    }
    require(
        cumulative <= Decimal(cost_summary["cumulative_authorized_maximum_usd"]),
        "actual cumulative cost exceeds authorization",
    )

    output = output.resolve(strict=False)
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.chmod(0o700)
    try:
        decision_fields = list(decisions[0])
        decisions_path = temporary / "api_cross_provider_decisions.csv"
        _write_csv(decisions_path, decisions, decision_fields)
        auto_path = temporary / "api_auto_accept_candidates.csv"
        _write_csv(auto_path, auto_accept, list(auto_accept[0]))

        queue_key = []
        packets: dict[int, list[dict[str, str]]] = {1: [], 2: []}
        for api_item_id in sorted(queue_ids):
            source = key_by_id[api_item_id]
            ids = {reviewer: _review_id(api_item_id, reviewer) for reviewer in (1, 2)}
            queue_key.append(
                {
                    "api_item_id": api_item_id,
                    "reviewer_1_item_id": ids[1],
                    "reviewer_2_item_id": ids[2],
                    "openalex_id": source["openalex_id"],
                    "episode_index": source["episode_index"],
                    "window_code": source["window_code"],
                    "source_row_number": source["source_row_number"],
                    "base_calibration": str(api_item_id in calibration_ids).lower(),
                    "api_escalated": str(api_item_id in escalation_ids).lower(),
                }
            )
            for reviewer in (1, 2):
                packets[reviewer].append(
                    {
                        "review_item_id": ids[reviewer],
                        "target_topic_name": source["target_topic_name"],
                        "title": source["title"],
                        "language": source["language"],
                        "work_type": source["work_type"],
                        "reviewer_label": "",
                        "reviewer_notes": "",
                    }
                )
        queue_key_path = temporary / "human_review_queue_key.csv"
        _write_csv(queue_key_path, queue_key, list(queue_key[0]))
        packet_paths = {}
        for reviewer in (1, 2):
            packet_rows = sorted(packets[reviewer], key=lambda row: row["review_item_id"])
            packet_path = temporary / f"human_review_reviewer_{reviewer}.csv"
            _write_csv(packet_path, packet_rows, PACKET_FIELDS)
            packet_paths[reviewer] = packet_path

        pair_counts = {
            f"{left}|{right}": count
            for (left, right), count in sorted(label_pairs.items())
        }
        confidence_pair_counts = {
            f"{left}|{right}": count
            for (left, right), count in sorted(confidence_pairs.items())
        }
        summary = {
            "schema_version": "pivot_feedback_d0_cross_provider_result_summary.v6",
            "status": STATUS,
            "provider_validation": {
                "openai_valid_structured_rows": 384,
                "claude_valid_structured_rows": 384,
                "shared_expected_custom_ids": 384,
                "provider_error_rows": 0,
                "incomplete_rows": 0,
            },
            "openai_label_counts": dict(sorted(openai_labels.items())),
            "claude_label_counts": dict(sorted(claude_labels.items())),
            "openai_confidence_counts": dict(sorted(openai_confidences.items())),
            "claude_confidence_counts": dict(sorted(claude_confidences.items())),
            "label_pair_counts": pair_counts,
            "confidence_pair_counts": confidence_pair_counts,
            "escalation_reason_counts": dict(sorted(reason_counts.items())),
            "reduction": reduction,
            "cost_control": cost_summary,
            "human_calibration_complete": False,
            "d0_complete": False,
            "outcomes_read": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        summary_path = temporary / "api_result_summary.json"
        _write_json(summary_path, summary)
        summary_md_path = temporary / "API_RESULT_SUMMARY.md"
        summary_md_path.write_text(
            "# D0 cross-provider API review v6\n\n"
            f"Status: `{STATUS}`\n\n"
            "- OpenAI retry: 384/384 valid structured responses.\n"
            "- Claude: 384/384 valid structured responses.\n"
            f"- Label agreement: {agreement}/384 "
            f"({Decimal(agreement * 100 / 384).quantize(Decimal('0.1'))}%).\n"
            f"- Dual HIGH-confidence ON_TOPIC auto-accept candidates: "
            f"{len(auto_accept)}.\n"
            f"- API escalation rows: {len(escalation_ids)}.\n"
            f"- Frozen calibration plus escalation union: {len(queue_ids)} rows "
            "in each of two blinded human packets.\n"
            f"- Cumulative API cost estimate, including the invalid v4 OpenAI "
            f"attempt: ${cost_summary['cumulative_actual_cost_usd']}.\n\n"
            "Human calibration and adjudication remain pending. D0, outcomes, "
            "G0, model fitting, G4, and any scientific effect remain closed.\n",
            encoding="utf-8",
        )
        summary_md_path.chmod(0o600)

        primary = [
            _artifact(decisions_path, rows=384),
            _artifact(auto_path, rows=len(auto_accept)),
            _artifact(queue_key_path, rows=len(queue_ids)),
            _artifact(packet_paths[1], rows=len(queue_ids)),
            _artifact(packet_paths[2], rows=len(queue_ids)),
            _artifact(summary_path),
            _artifact(summary_md_path),
        ]
        manifest_path = temporary / "result_manifest.json"
        manifest = {
            "schema_version": "pivot_feedback_d0_cross_provider_result_manifest.v6",
            "status": STATUS,
            "config_relative_path": config_path.relative_to(repo_root).as_posix(),
            "config_sha256": _sha256(config_path),
            "input_sha256": {
                path.name: expected_sha for path, expected_sha in bindings.items()
            },
            "reduction": reduction,
            "cost_control": cost_summary,
            "artifacts": primary,
            "human_packets_blinded_from_api_outputs": True,
            "human_calibration_complete": False,
            "d0_complete": False,
            "outcomes_read": False,
        }
        _write_json(manifest_path, manifest)
        artifacts = [*primary, _artifact(manifest_path)]
        receipt = {
            "schema_version": "pivot_feedback_d0_cross_provider_result_outputs.v6",
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": _artifact_tree(artifacts),
            "artifacts": artifacts,
            "openai_valid_structured_rows": 384,
            "claude_valid_structured_rows": 384,
            "shared_expected_custom_ids": 384,
            "reduction": reduction,
            "cost_control": cost_summary,
            "human_packets_blinded_from_api_outputs": True,
            "reviewers_required": 2,
            "human_calibration_complete": False,
            "human_adjudication_complete": False,
            "candidate_universe_screen_complete": False,
            "episode_3_quarantined": True,
            "d0_complete": False,
            "outcomes_read": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        _write_json(temporary / "cross_provider_result.outputs.json", receipt)
        require(
            all(path.stat().st_mode & 0o777 == 0o600 for path in temporary.iterdir()),
            "file mode differs",
        )
        os.replace(temporary, output)
        temporary = None
        return receipt
    finally:
        _cleanup_temporary_directory(temporary, output=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--v4-prepared-root",
        type=Path,
        default=DEFAULT_V4_PREPARED_ROOT,
    )
    parser.add_argument(
        "--v4-execution-root",
        type=Path,
        default=DEFAULT_V4_EXECUTION_ROOT,
    )
    parser.add_argument(
        "--v5-prepared-root",
        type=Path,
        default=DEFAULT_V5_PREPARED_ROOT,
    )
    parser.add_argument("--v5-report", type=Path, default=DEFAULT_V5_REPORT)
    parser.add_argument(
        "--v5-execution-root",
        type=Path,
        default=DEFAULT_V5_EXECUTION_ROOT,
    )
    parser.add_argument(
        "--terminal-snapshot",
        type=Path,
        default=(
            DEFAULT_V5_EXECUTION_ROOT
            / "status_snapshots/status-20260729T050931994847Z.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(
            config_path=args.config,
            v4_prepared_root=args.v4_prepared_root,
            v4_execution_root=args.v4_execution_root,
            v5_prepared_root=args.v5_prepared_root,
            v5_report=args.v5_report,
            v5_execution_root=args.v5_execution_root,
            terminal_snapshot=args.terminal_snapshot,
            output=args.output,
        )
        print(result["status"])
        print(f"output={args.output}")
        print(
            "human_review_rows="
            f"{result['reduction']['blinded_human_queue_rows']}"
        )
        print(
            "cumulative_actual_cost_usd="
            f"{result['cost_control']['cumulative_actual_cost_usd']}"
        )
        return 0
    except (
        CrossProviderResultBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
